import sys
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type as TypingType, Union
from mypy.errorcodes import ErrorCode
from mypy.nodes import (
from mypy.options import Options
from mypy.plugin import (
from mypy.plugins import dataclasses
from mypy.semanal import set_callable_name  # type: ignore
from mypy.server.trigger import make_wildcard_trigger
from mypy.types import (
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic.utils import is_valid_field
class PydanticModelTransformer:
    tracked_config_fields: Set[str] = {'extra', 'allow_mutation', 'frozen', 'orm_mode', 'allow_population_by_field_name', 'alias_generator'}

    def __init__(self, ctx: ClassDefContext, plugin_config: PydanticPluginConfig) -> None:
        self._ctx = ctx
        self.plugin_config = plugin_config

    def transform(self) -> None:
        """
        Configures the BaseModel subclass according to the plugin settings.

        In particular:
        * determines the model config and fields,
        * adds a fields-aware signature for the initializer and construct methods
        * freezes the class if allow_mutation = False or frozen = True
        * stores the fields, config, and if the class is settings in the mypy metadata for access by subclasses
        """
        ctx = self._ctx
        info = ctx.cls.info
        self.adjust_validator_signatures()
        config = self.collect_config()
        fields = self.collect_fields(config)
        is_settings = any((get_fullname(base) == BASESETTINGS_FULLNAME for base in info.mro[:-1]))
        self.add_initializer(fields, config, is_settings)
        self.add_construct_method(fields)
        self.set_frozen(fields, frozen=config.allow_mutation is False or config.frozen is True)
        info.metadata[METADATA_KEY] = {'fields': {field.name: field.serialize() for field in fields}, 'config': config.set_values_dict()}

    def adjust_validator_signatures(self) -> None:
        """When we decorate a function `f` with `pydantic.validator(...), mypy sees
        `f` as a regular method taking a `self` instance, even though pydantic
        internally wraps `f` with `classmethod` if necessary.

        Teach mypy this by marking any function whose outermost decorator is a
        `validator()` call as a classmethod.
        """
        for name, sym in self._ctx.cls.info.names.items():
            if isinstance(sym.node, Decorator):
                first_dec = sym.node.original_decorators[0]
                if isinstance(first_dec, CallExpr) and isinstance(first_dec.callee, NameExpr) and (first_dec.callee.fullname == f'{_NAMESPACE}.class_validators.validator'):
                    sym.node.func.is_class = True

    def collect_config(self) -> 'ModelConfigData':
        """
        Collects the values of the config attributes that are used by the plugin, accounting for parent classes.
        """
        ctx = self._ctx
        cls = ctx.cls
        config = ModelConfigData()
        for stmt in cls.defs.body:
            if not isinstance(stmt, ClassDef):
                continue
            if stmt.name == 'Config':
                for substmt in stmt.defs.body:
                    if not isinstance(substmt, AssignmentStmt):
                        continue
                    config.update(self.get_config_update(substmt))
                if config.has_alias_generator and (not config.allow_population_by_field_name) and self.plugin_config.warn_required_dynamic_aliases:
                    error_required_dynamic_aliases(ctx.api, stmt)
        for info in cls.info.mro[1:]:
            if METADATA_KEY not in info.metadata:
                continue
            ctx.api.add_plugin_dependency(make_wildcard_trigger(get_fullname(info)))
            for name, value in info.metadata[METADATA_KEY]['config'].items():
                config.setdefault(name, value)
        return config

    def collect_fields(self, model_config: 'ModelConfigData') -> List['PydanticModelField']:
        """
        Collects the fields for the model, accounting for parent classes
        """
        ctx = self._ctx
        cls = self._ctx.cls
        fields = []
        known_fields = set()
        for stmt in cls.defs.body:
            if not isinstance(stmt, AssignmentStmt):
                continue
            lhs = stmt.lvalues[0]
            if not isinstance(lhs, NameExpr) or not is_valid_field(lhs.name):
                continue
            if not stmt.new_syntax and self.plugin_config.warn_untyped_fields:
                error_untyped_fields(ctx.api, stmt)
            sym = cls.info.names.get(lhs.name)
            if sym is None:
                continue
            node = sym.node
            if isinstance(node, PlaceholderNode):
                continue
            if not isinstance(node, Var):
                continue
            if node.is_classvar:
                continue
            is_required = self.get_is_required(cls, stmt, lhs)
            alias, has_dynamic_alias = self.get_alias_info(stmt)
            if has_dynamic_alias and (not model_config.allow_population_by_field_name) and self.plugin_config.warn_required_dynamic_aliases:
                error_required_dynamic_aliases(ctx.api, stmt)
            fields.append(PydanticModelField(name=lhs.name, is_required=is_required, alias=alias, has_dynamic_alias=has_dynamic_alias, line=stmt.line, column=stmt.column))
            known_fields.add(lhs.name)
        all_fields = fields.copy()
        for info in cls.info.mro[1:]:
            if METADATA_KEY not in info.metadata:
                continue
            superclass_fields = []
            ctx.api.add_plugin_dependency(make_wildcard_trigger(get_fullname(info)))
            for name, data in info.metadata[METADATA_KEY]['fields'].items():
                if name not in known_fields:
                    field = PydanticModelField.deserialize(info, data)
                    known_fields.add(name)
                    superclass_fields.append(field)
                else:
                    field, = (a for a in all_fields if a.name == name)
                    all_fields.remove(field)
                    superclass_fields.append(field)
            all_fields = superclass_fields + all_fields
        return all_fields

    def add_initializer(self, fields: List['PydanticModelField'], config: 'ModelConfigData', is_settings: bool) -> None:
        """
        Adds a fields-aware `__init__` method to the class.

        The added `__init__` will be annotated with types vs. all `Any` depending on the plugin settings.
        """
        ctx = self._ctx
        typed = self.plugin_config.init_typed
        use_alias = config.allow_population_by_field_name is not True
        force_all_optional = is_settings or bool(config.has_alias_generator and (not config.allow_population_by_field_name))
        init_arguments = self.get_field_arguments(fields, typed=typed, force_all_optional=force_all_optional, use_alias=use_alias)
        if not self.should_init_forbid_extra(fields, config):
            var = Var('kwargs')
            init_arguments.append(Argument(var, AnyType(TypeOfAny.explicit), None, ARG_STAR2))
        if '__init__' not in ctx.cls.info.names:
            add_method(ctx, '__init__', init_arguments, NoneType())

    def add_construct_method(self, fields: List['PydanticModelField']) -> None:
        """
        Adds a fully typed `construct` classmethod to the class.

        Similar to the fields-aware __init__ method, but always uses the field names (not aliases),
        and does not treat settings fields as optional.
        """
        ctx = self._ctx
        set_str = ctx.api.named_type(f'{BUILTINS_NAME}.set', [ctx.api.named_type(f'{BUILTINS_NAME}.str')])
        optional_set_str = UnionType([set_str, NoneType()])
        fields_set_argument = Argument(Var('_fields_set', optional_set_str), optional_set_str, None, ARG_OPT)
        construct_arguments = self.get_field_arguments(fields, typed=True, force_all_optional=False, use_alias=False)
        construct_arguments = [fields_set_argument] + construct_arguments
        obj_type = ctx.api.named_type(f'{BUILTINS_NAME}.object')
        self_tvar_name = '_PydanticBaseModel'
        tvar_fullname = ctx.cls.fullname + '.' + self_tvar_name
        if MYPY_VERSION_TUPLE >= (1, 4):
            tvd = TypeVarType(self_tvar_name, tvar_fullname, -1, [], obj_type, AnyType(TypeOfAny.from_omitted_generics))
            self_tvar_expr = TypeVarExpr(self_tvar_name, tvar_fullname, [], obj_type, AnyType(TypeOfAny.from_omitted_generics))
        else:
            tvd = TypeVarDef(self_tvar_name, tvar_fullname, -1, [], obj_type)
            self_tvar_expr = TypeVarExpr(self_tvar_name, tvar_fullname, [], obj_type)
        ctx.cls.info.names[self_tvar_name] = SymbolTableNode(MDEF, self_tvar_expr)
        if isinstance(tvd, TypeVarType):
            self_type = tvd
        else:
            self_type = TypeVarType(tvd)
        add_method(ctx, 'construct', construct_arguments, return_type=self_type, self_type=self_type, tvar_def=tvd, is_classmethod=True)

    def set_frozen(self, fields: List['PydanticModelField'], frozen: bool) -> None:
        """
        Marks all fields as properties so that attempts to set them trigger mypy errors.

        This is the same approach used by the attrs and dataclasses plugins.
        """
        ctx = self._ctx
        info = ctx.cls.info
        for field in fields:
            sym_node = info.names.get(field.name)
            if sym_node is not None:
                var = sym_node.node
                if isinstance(var, Var):
                    var.is_property = frozen
                elif isinstance(var, PlaceholderNode) and (not ctx.api.final_iteration):
                    ctx.api.defer()
                else:
                    try:
                        var_str = str(var)
                    except TypeError:
                        var_str = repr(var)
                    detail = f'sym_node.node: {var_str} (of type {var.__class__})'
                    error_unexpected_behavior(detail, ctx.api, ctx.cls)
            else:
                var = field.to_var(info, use_alias=False)
                var.info = info
                var.is_property = frozen
                var._fullname = get_fullname(info) + '.' + get_name(var)
                info.names[get_name(var)] = SymbolTableNode(MDEF, var)

    def get_config_update(self, substmt: AssignmentStmt) -> Optional['ModelConfigData']:
        """
        Determines the config update due to a single statement in the Config class definition.

        Warns if a tracked config attribute is set to a value the plugin doesn't know how to interpret (e.g., an int)
        """
        lhs = substmt.lvalues[0]
        if not (isinstance(lhs, NameExpr) and lhs.name in self.tracked_config_fields):
            return None
        if lhs.name == 'extra':
            if isinstance(substmt.rvalue, StrExpr):
                forbid_extra = substmt.rvalue.value == 'forbid'
            elif isinstance(substmt.rvalue, MemberExpr):
                forbid_extra = substmt.rvalue.name == 'forbid'
            else:
                error_invalid_config_value(lhs.name, self._ctx.api, substmt)
                return None
            return ModelConfigData(forbid_extra=forbid_extra)
        if lhs.name == 'alias_generator':
            has_alias_generator = True
            if isinstance(substmt.rvalue, NameExpr) and substmt.rvalue.fullname == 'builtins.None':
                has_alias_generator = False
            return ModelConfigData(has_alias_generator=has_alias_generator)
        if isinstance(substmt.rvalue, NameExpr) and substmt.rvalue.fullname in ('builtins.True', 'builtins.False'):
            return ModelConfigData(**{lhs.name: substmt.rvalue.fullname == 'builtins.True'})
        error_invalid_config_value(lhs.name, self._ctx.api, substmt)
        return None

    @staticmethod
    def get_is_required(cls: ClassDef, stmt: AssignmentStmt, lhs: NameExpr) -> bool:
        """
        Returns a boolean indicating whether the field defined in `stmt` is a required field.
        """
        expr = stmt.rvalue
        if isinstance(expr, TempNode):
            value_type = get_proper_type(cls.info[lhs.name].type)
            return not PydanticModelTransformer.type_has_implicit_default(value_type)
        if isinstance(expr, CallExpr) and isinstance(expr.callee, RefExpr) and (expr.callee.fullname == FIELD_FULLNAME):
            for arg, name in zip(expr.args, expr.arg_names):
                if name is None or name == 'default':
                    return arg.__class__ is EllipsisExpr
                if name == 'default_factory':
                    return False
            value_type = get_proper_type(cls.info[lhs.name].type)
            return not PydanticModelTransformer.type_has_implicit_default(value_type)
        return isinstance(expr, EllipsisExpr)

    @staticmethod
    def type_has_implicit_default(type_: Optional[ProperType]) -> bool:
        """
        Returns True if the passed type will be given an implicit default value.

        In pydantic v1, this is the case for Optional types and Any (with default value None).
        """
        if isinstance(type_, AnyType):
            return True
        if isinstance(type_, UnionType) and any((isinstance(item, NoneType) or isinstance(item, AnyType) for item in type_.items)):
            return True
        return False

    @staticmethod
    def get_alias_info(stmt: AssignmentStmt) -> Tuple[Optional[str], bool]:
        """
        Returns a pair (alias, has_dynamic_alias), extracted from the declaration of the field defined in `stmt`.

        `has_dynamic_alias` is True if and only if an alias is provided, but not as a string literal.
        If `has_dynamic_alias` is True, `alias` will be None.
        """
        expr = stmt.rvalue
        if isinstance(expr, TempNode):
            return (None, False)
        if not (isinstance(expr, CallExpr) and isinstance(expr.callee, RefExpr) and (expr.callee.fullname == FIELD_FULLNAME)):
            return (None, False)
        for i, arg_name in enumerate(expr.arg_names):
            if arg_name != 'alias':
                continue
            arg = expr.args[i]
            if isinstance(arg, StrExpr):
                return (arg.value, False)
            else:
                return (None, True)
        return (None, False)

    def get_field_arguments(self, fields: List['PydanticModelField'], typed: bool, force_all_optional: bool, use_alias: bool) -> List[Argument]:
        """
        Helper function used during the construction of the `__init__` and `construct` method signatures.

        Returns a list of mypy Argument instances for use in the generated signatures.
        """
        info = self._ctx.cls.info
        arguments = [field.to_argument(info, typed=typed, force_optional=force_all_optional, use_alias=use_alias) for field in fields if not (use_alias and field.has_dynamic_alias)]
        return arguments

    def should_init_forbid_extra(self, fields: List['PydanticModelField'], config: 'ModelConfigData') -> bool:
        """
        Indicates whether the generated `__init__` should get a `**kwargs` at the end of its signature

        We disallow arbitrary kwargs if the extra config setting is "forbid", or if the plugin config says to,
        *unless* a required dynamic alias is present (since then we can't determine a valid signature).
        """
        if not config.allow_population_by_field_name:
            if self.is_dynamic_alias_present(fields, bool(config.has_alias_generator)):
                return False
        if config.forbid_extra:
            return True
        return self.plugin_config.init_forbid_extra

    @staticmethod
    def is_dynamic_alias_present(fields: List['PydanticModelField'], has_alias_generator: bool) -> bool:
        """
        Returns whether any fields on the model have a "dynamic alias", i.e., an alias that cannot be
        determined during static analysis.
        """
        for field in fields:
            if field.has_dynamic_alias:
                return True
        if has_alias_generator:
            for field in fields:
                if field.alias is None:
                    return True
        return False