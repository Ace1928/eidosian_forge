from __future__ import annotations as _annotations
import collections.abc
import dataclasses
import inspect
import re
import sys
import typing
import warnings
from contextlib import contextmanager
from copy import copy, deepcopy
from enum import Enum
from functools import partial
from inspect import Parameter, _ParameterKind, signature
from itertools import chain
from operator import attrgetter
from types import FunctionType, LambdaType, MethodType
from typing import (
from warnings import warn
from pydantic_core import CoreSchema, PydanticUndefined, core_schema, to_jsonable_python
from typing_extensions import Annotated, Literal, TypeAliasType, TypedDict, get_args, get_origin, is_typeddict
from ..aliases import AliasGenerator
from ..annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from ..config import ConfigDict, JsonDict, JsonEncoder
from ..errors import PydanticSchemaGenerationError, PydanticUndefinedAnnotation, PydanticUserError
from ..json_schema import JsonSchemaValue
from ..version import version_short
from ..warnings import PydanticDeprecatedSince20
from . import _core_utils, _decorators, _discriminated_union, _known_annotated_metadata, _typing_extra
from ._config import ConfigWrapper, ConfigWrapperStack
from ._core_metadata import CoreMetadataHandler, build_metadata_dict
from ._core_utils import (
from ._decorators import (
from ._fields import collect_dataclass_fields, get_type_hints_infer_globalns
from ._forward_ref import PydanticRecursiveRef
from ._generics import get_standard_typevars_map, has_instance_in_type, recursively_defined_type_refs, replace_types
from ._schema_generation_shared import (
from ._typing_extra import is_finalvar
from ._utils import lenient_issubclass
class GenerateSchema:
    """Generate core schema for a Pydantic model, dataclass and types like `str`, `datetime`, ... ."""
    __slots__ = ('_config_wrapper_stack', '_types_namespace_stack', '_typevars_map', '_has_invalid_schema', 'field_name_stack', 'defs')

    def __init__(self, config_wrapper: ConfigWrapper, types_namespace: dict[str, Any] | None, typevars_map: dict[Any, Any] | None=None) -> None:
        self._config_wrapper_stack = ConfigWrapperStack(config_wrapper)
        self._types_namespace_stack = TypesNamespaceStack(types_namespace)
        self._typevars_map = typevars_map
        self._has_invalid_schema = False
        self.field_name_stack = _FieldNameStack()
        self.defs = _Definitions()

    @classmethod
    def __from_parent(cls, config_wrapper_stack: ConfigWrapperStack, types_namespace_stack: TypesNamespaceStack, typevars_map: dict[Any, Any] | None, defs: _Definitions) -> GenerateSchema:
        obj = cls.__new__(cls)
        obj._config_wrapper_stack = config_wrapper_stack
        obj._types_namespace_stack = types_namespace_stack
        obj._typevars_map = typevars_map
        obj._has_invalid_schema = False
        obj.field_name_stack = _FieldNameStack()
        obj.defs = defs
        return obj

    @property
    def _config_wrapper(self) -> ConfigWrapper:
        return self._config_wrapper_stack.tail

    @property
    def _types_namespace(self) -> dict[str, Any] | None:
        return self._types_namespace_stack.tail

    @property
    def _current_generate_schema(self) -> GenerateSchema:
        cls = self._config_wrapper.schema_generator or GenerateSchema
        return cls.__from_parent(self._config_wrapper_stack, self._types_namespace_stack, self._typevars_map, self.defs)

    @property
    def _arbitrary_types(self) -> bool:
        return self._config_wrapper.arbitrary_types_allowed

    def str_schema(self) -> CoreSchema:
        """Generate a CoreSchema for `str`"""
        return core_schema.str_schema()

    def _list_schema(self, tp: Any, items_type: Any) -> CoreSchema:
        return core_schema.list_schema(self.generate_schema(items_type))

    def _dict_schema(self, tp: Any, keys_type: Any, values_type: Any) -> CoreSchema:
        return core_schema.dict_schema(self.generate_schema(keys_type), self.generate_schema(values_type))

    def _set_schema(self, tp: Any, items_type: Any) -> CoreSchema:
        return core_schema.set_schema(self.generate_schema(items_type))

    def _frozenset_schema(self, tp: Any, items_type: Any) -> CoreSchema:
        return core_schema.frozenset_schema(self.generate_schema(items_type))

    def _arbitrary_type_schema(self, tp: Any) -> CoreSchema:
        if not isinstance(tp, type):
            warn(f'{tp!r} is not a Python type (it may be an instance of an object), Pydantic will allow any object with no validation since we cannot even enforce that the input is an instance of the given type. To get rid of this error wrap the type with `pydantic.SkipValidation`.', UserWarning)
            return core_schema.any_schema()
        return core_schema.is_instance_schema(tp)

    def _unknown_type_schema(self, obj: Any) -> CoreSchema:
        raise PydanticSchemaGenerationError(f'Unable to generate pydantic-core schema for {obj!r}. Set `arbitrary_types_allowed=True` in the model_config to ignore this error or implement `__get_pydantic_core_schema__` on your type to fully support it.\n\nIf you got this error by calling handler(<some type>) within `__get_pydantic_core_schema__` then you likely need to call `handler.generate_schema(<some type>)` since we do not call `__get_pydantic_core_schema__` on `<some type>` otherwise to avoid infinite recursion.')

    def _apply_discriminator_to_union(self, schema: CoreSchema, discriminator: str | Discriminator | None) -> CoreSchema:
        if discriminator is None:
            return schema
        try:
            return _discriminated_union.apply_discriminator(schema, discriminator)
        except _discriminated_union.MissingDefinitionForUnionRef:
            _discriminated_union.set_discriminator_in_metadata(schema, discriminator)
            return schema

    class CollectedInvalid(Exception):
        pass

    def clean_schema(self, schema: CoreSchema) -> CoreSchema:
        schema = self.collect_definitions(schema)
        schema = simplify_schema_references(schema)
        schema = _discriminated_union.apply_discriminators(schema)
        if collect_invalid_schemas(schema):
            raise self.CollectedInvalid()
        schema = validate_core_schema(schema)
        return schema

    def collect_definitions(self, schema: CoreSchema) -> CoreSchema:
        ref = cast('str | None', schema.get('ref', None))
        if ref:
            self.defs.definitions[ref] = schema
        if 'ref' in schema:
            schema = core_schema.definition_reference_schema(schema['ref'])
        return core_schema.definitions_schema(schema, list(self.defs.definitions.values()))

    def _add_js_function(self, metadata_schema: CoreSchema, js_function: Callable[..., Any]) -> None:
        metadata = CoreMetadataHandler(metadata_schema).metadata
        pydantic_js_functions = metadata.setdefault('pydantic_js_functions', [])
        if js_function not in pydantic_js_functions:
            pydantic_js_functions.append(js_function)

    def generate_schema(self, obj: Any, from_dunder_get_core_schema: bool=True) -> core_schema.CoreSchema:
        """Generate core schema.

        Args:
            obj: The object to generate core schema for.
            from_dunder_get_core_schema: Whether to generate schema from either the
                `__get_pydantic_core_schema__` function or `__pydantic_core_schema__` property.

        Returns:
            The generated core schema.

        Raises:
            PydanticUndefinedAnnotation:
                If it is not possible to evaluate forward reference.
            PydanticSchemaGenerationError:
                If it is not possible to generate pydantic-core schema.
            TypeError:
                - If `alias_generator` returns a disallowed type (must be str, AliasPath or AliasChoices).
                - If V1 style validator with `each_item=True` applied on a wrong field.
            PydanticUserError:
                - If `typing.TypedDict` is used instead of `typing_extensions.TypedDict` on Python < 3.12.
                - If `__modify_schema__` method is used instead of `__get_pydantic_json_schema__`.
        """
        schema: CoreSchema | None = None
        if from_dunder_get_core_schema:
            from_property = self._generate_schema_from_property(obj, obj)
            if from_property is not None:
                schema = from_property
        if schema is None:
            schema = self._generate_schema(obj)
        metadata_js_function = _extract_get_pydantic_json_schema(obj, schema)
        if metadata_js_function is not None:
            metadata_schema = resolve_original_schema(schema, self.defs.definitions)
            if metadata_schema:
                self._add_js_function(metadata_schema, metadata_js_function)
        schema = _add_custom_serialization_from_json_encoders(self._config_wrapper.json_encoders, obj, schema)
        schema = self._post_process_generated_schema(schema)
        return schema

    def _model_schema(self, cls: type[BaseModel]) -> core_schema.CoreSchema:
        """Generate schema for a Pydantic model."""
        with self.defs.get_schema_or_ref(cls) as (model_ref, maybe_schema):
            if maybe_schema is not None:
                return maybe_schema
            fields = cls.model_fields
            decorators = cls.__pydantic_decorators__
            computed_fields = decorators.computed_fields
            check_decorator_fields_exist(chain(decorators.field_validators.values(), decorators.field_serializers.values(), decorators.validators.values()), {*fields.keys(), *computed_fields.keys()})
            config_wrapper = ConfigWrapper(cls.model_config, check=False)
            core_config = config_wrapper.core_config(cls)
            metadata = build_metadata_dict(js_functions=[partial(modify_model_json_schema, cls=cls)])
            model_validators = decorators.model_validators.values()
            extras_schema = None
            if core_config.get('extra_fields_behavior') == 'allow':
                for tp in (cls, *cls.__mro__):
                    extras_annotation = cls.__annotations__.get('__pydantic_extra__', None)
                    if extras_annotation is not None:
                        tp = get_origin(extras_annotation)
                        if tp not in (Dict, dict):
                            raise PydanticSchemaGenerationError('The type annotation for `__pydantic_extra__` must be `Dict[str, ...]`')
                        extra_items_type = self._get_args_resolving_forward_refs(cls.__annotations__['__pydantic_extra__'], required=True)[1]
                        if extra_items_type is not Any:
                            extras_schema = self.generate_schema(extra_items_type)
                            break
            with self._config_wrapper_stack.push(config_wrapper), self._types_namespace_stack.push(cls):
                self = self._current_generate_schema
                if cls.__pydantic_root_model__:
                    root_field = self._common_field_schema('root', fields['root'], decorators)
                    inner_schema = root_field['schema']
                    inner_schema = apply_model_validators(inner_schema, model_validators, 'inner')
                    model_schema = core_schema.model_schema(cls, inner_schema, custom_init=getattr(cls, '__pydantic_custom_init__', None), root_model=True, post_init=getattr(cls, '__pydantic_post_init__', None), config=core_config, ref=model_ref, metadata=metadata)
                else:
                    fields_schema: core_schema.CoreSchema = core_schema.model_fields_schema({k: self._generate_md_field_schema(k, v, decorators) for k, v in fields.items()}, computed_fields=[self._computed_field_schema(d, decorators.field_serializers) for d in computed_fields.values()], extras_schema=extras_schema, model_name=cls.__name__)
                    inner_schema = apply_validators(fields_schema, decorators.root_validators.values(), None)
                    new_inner_schema = define_expected_missing_refs(inner_schema, recursively_defined_type_refs())
                    if new_inner_schema is not None:
                        inner_schema = new_inner_schema
                    inner_schema = apply_model_validators(inner_schema, model_validators, 'inner')
                    model_schema = core_schema.model_schema(cls, inner_schema, custom_init=getattr(cls, '__pydantic_custom_init__', None), root_model=False, post_init=getattr(cls, '__pydantic_post_init__', None), config=core_config, ref=model_ref, metadata=metadata)
                schema = self._apply_model_serializers(model_schema, decorators.model_serializers.values())
                schema = apply_model_validators(schema, model_validators, 'outer')
                self.defs.definitions[model_ref] = self._post_process_generated_schema(schema)
                return core_schema.definition_reference_schema(model_ref)

    def _unpack_refs_defs(self, schema: CoreSchema) -> CoreSchema:
        """Unpack all 'definitions' schemas into `GenerateSchema.defs.definitions`
        and return the inner schema.
        """

        def get_ref(s: CoreSchema) -> str:
            return s['ref']
        if schema['type'] == 'definitions':
            self.defs.definitions.update({get_ref(s): s for s in schema['definitions']})
            schema = schema['schema']
        return schema

    def _generate_schema_from_property(self, obj: Any, source: Any) -> core_schema.CoreSchema | None:
        """Try to generate schema from either the `__get_pydantic_core_schema__` function or
        `__pydantic_core_schema__` property.

        Note: `__get_pydantic_core_schema__` takes priority so it can
        decide whether to use a `__pydantic_core_schema__` attribute, or generate a fresh schema.
        """
        with self.defs.get_schema_or_ref(obj) as (_, maybe_schema):
            if maybe_schema is not None:
                return maybe_schema
        if obj is source:
            ref_mode = 'unpack'
        else:
            ref_mode = 'to-def'
        schema: CoreSchema
        get_schema = getattr(obj, '__get_pydantic_core_schema__', None)
        if get_schema is None:
            validators = getattr(obj, '__get_validators__', None)
            if validators is None:
                return None
            warn('`__get_validators__` is deprecated and will be removed, use `__get_pydantic_core_schema__` instead.', PydanticDeprecatedSince20)
            schema = core_schema.chain_schema([core_schema.with_info_plain_validator_function(v) for v in validators()])
        elif len(inspect.signature(get_schema).parameters) == 1:
            schema = get_schema(source)
        else:
            schema = get_schema(source, CallbackGetCoreSchemaHandler(self._generate_schema, self, ref_mode=ref_mode))
        schema = self._unpack_refs_defs(schema)
        if is_function_with_inner_schema(schema):
            ref = schema['schema'].pop('ref', None)
            if ref:
                schema['ref'] = ref
        else:
            ref = get_ref(schema)
        if ref:
            self.defs.definitions[ref] = self._post_process_generated_schema(schema)
            return core_schema.definition_reference_schema(ref)
        schema = self._post_process_generated_schema(schema)
        return schema

    def _resolve_forward_ref(self, obj: Any) -> Any:
        try:
            obj = _typing_extra.eval_type_backport(obj, globalns=self._types_namespace)
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        if isinstance(obj, ForwardRef):
            raise PydanticUndefinedAnnotation(obj.__forward_arg__, f'Unable to evaluate forward reference {obj}')
        if self._typevars_map:
            obj = replace_types(obj, self._typevars_map)
        return obj

    @overload
    def _get_args_resolving_forward_refs(self, obj: Any, required: Literal[True]) -> tuple[Any, ...]:
        ...

    @overload
    def _get_args_resolving_forward_refs(self, obj: Any) -> tuple[Any, ...] | None:
        ...

    def _get_args_resolving_forward_refs(self, obj: Any, required: bool=False) -> tuple[Any, ...] | None:
        args = get_args(obj)
        if args:
            args = tuple([self._resolve_forward_ref(a) if isinstance(a, ForwardRef) else a for a in args])
        elif required:
            raise TypeError(f'Expected {obj} to have generic parameters but it had none')
        return args

    def _get_first_arg_or_any(self, obj: Any) -> Any:
        args = self._get_args_resolving_forward_refs(obj)
        if not args:
            return Any
        return args[0]

    def _get_first_two_args_or_any(self, obj: Any) -> tuple[Any, Any]:
        args = self._get_args_resolving_forward_refs(obj)
        if not args:
            return (Any, Any)
        if len(args) < 2:
            origin = get_origin(obj)
            raise TypeError(f'Expected two type arguments for {origin}, got 1')
        return (args[0], args[1])

    def _post_process_generated_schema(self, schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
        if 'metadata' not in schema:
            schema['metadata'] = {}
        return schema

    def _generate_schema(self, obj: Any) -> core_schema.CoreSchema:
        """Recursively generate a pydantic-core schema for any supported python type."""
        has_invalid_schema = self._has_invalid_schema
        self._has_invalid_schema = False
        schema = self._generate_schema_inner(obj)
        self._has_invalid_schema = self._has_invalid_schema or has_invalid_schema
        return schema

    def _generate_schema_inner(self, obj: Any) -> core_schema.CoreSchema:
        if isinstance(obj, _AnnotatedType):
            return self._annotated_schema(obj)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            obj = ForwardRef(obj)
        if isinstance(obj, ForwardRef):
            return self.generate_schema(self._resolve_forward_ref(obj))
        from ..main import BaseModel
        if lenient_issubclass(obj, BaseModel):
            return self._model_schema(obj)
        if isinstance(obj, PydanticRecursiveRef):
            return core_schema.definition_reference_schema(schema_ref=obj.type_ref)
        return self.match_type(obj)

    def match_type(self, obj: Any) -> core_schema.CoreSchema:
        """Main mapping of types to schemas.

        The general structure is a series of if statements starting with the simple cases
        (non-generic primitive types) and then handling generics and other more complex cases.

        Each case either generates a schema directly, calls into a public user-overridable method
        (like `GenerateSchema.tuple_variable_schema`) or calls into a private method that handles some
        boilerplate before calling into the user-facing method (e.g. `GenerateSchema._tuple_schema`).

        The idea is that we'll evolve this into adding more and more user facing methods over time
        as they get requested and we figure out what the right API for them is.
        """
        if obj is str:
            return self.str_schema()
        elif obj is bytes:
            return core_schema.bytes_schema()
        elif obj is int:
            return core_schema.int_schema()
        elif obj is float:
            return core_schema.float_schema()
        elif obj is bool:
            return core_schema.bool_schema()
        elif obj is Any or obj is object:
            return core_schema.any_schema()
        elif obj is None or obj is _typing_extra.NoneType:
            return core_schema.none_schema()
        elif obj in TUPLE_TYPES:
            return self._tuple_schema(obj)
        elif obj in LIST_TYPES:
            return self._list_schema(obj, self._get_first_arg_or_any(obj))
        elif obj in SET_TYPES:
            return self._set_schema(obj, self._get_first_arg_or_any(obj))
        elif obj in FROZEN_SET_TYPES:
            return self._frozenset_schema(obj, self._get_first_arg_or_any(obj))
        elif obj in DICT_TYPES:
            return self._dict_schema(obj, *self._get_first_two_args_or_any(obj))
        elif isinstance(obj, TypeAliasType):
            return self._type_alias_type_schema(obj)
        elif obj == type:
            return self._type_schema()
        elif _typing_extra.is_callable_type(obj):
            return core_schema.callable_schema()
        elif _typing_extra.is_literal_type(obj):
            return self._literal_schema(obj)
        elif is_typeddict(obj):
            return self._typed_dict_schema(obj, None)
        elif _typing_extra.is_namedtuple(obj):
            return self._namedtuple_schema(obj, None)
        elif _typing_extra.is_new_type(obj):
            return self.generate_schema(obj.__supertype__)
        elif obj == re.Pattern:
            return self._pattern_schema(obj)
        elif obj is collections.abc.Hashable or obj is typing.Hashable:
            return self._hashable_schema()
        elif isinstance(obj, typing.TypeVar):
            return self._unsubstituted_typevar_schema(obj)
        elif is_finalvar(obj):
            if obj is Final:
                return core_schema.any_schema()
            return self.generate_schema(self._get_first_arg_or_any(obj))
        elif isinstance(obj, (FunctionType, LambdaType, MethodType, partial)):
            return self._callable_schema(obj)
        elif inspect.isclass(obj) and issubclass(obj, Enum):
            from ._std_types_schema import get_enum_core_schema
            return get_enum_core_schema(obj, self._config_wrapper.config_dict)
        if _typing_extra.is_dataclass(obj):
            return self._dataclass_schema(obj, None)
        res = self._get_prepare_pydantic_annotations_for_known_type(obj, ())
        if res is not None:
            source_type, annotations = res
            return self._apply_annotations(source_type, annotations)
        origin = get_origin(obj)
        if origin is not None:
            return self._match_generic_type(obj, origin)
        if self._arbitrary_types:
            return self._arbitrary_type_schema(obj)
        return self._unknown_type_schema(obj)

    def _match_generic_type(self, obj: Any, origin: Any) -> CoreSchema:
        if isinstance(origin, TypeAliasType):
            return self._type_alias_type_schema(obj)
        if _typing_extra.is_dataclass(origin):
            return self._dataclass_schema(obj, origin)
        if _typing_extra.is_namedtuple(origin):
            return self._namedtuple_schema(obj, origin)
        from_property = self._generate_schema_from_property(origin, obj)
        if from_property is not None:
            return from_property
        if _typing_extra.origin_is_union(origin):
            return self._union_schema(obj)
        elif origin in TUPLE_TYPES:
            return self._tuple_schema(obj)
        elif origin in LIST_TYPES:
            return self._list_schema(obj, self._get_first_arg_or_any(obj))
        elif origin in SET_TYPES:
            return self._set_schema(obj, self._get_first_arg_or_any(obj))
        elif origin in FROZEN_SET_TYPES:
            return self._frozenset_schema(obj, self._get_first_arg_or_any(obj))
        elif origin in DICT_TYPES:
            return self._dict_schema(obj, *self._get_first_two_args_or_any(obj))
        elif is_typeddict(origin):
            return self._typed_dict_schema(obj, origin)
        elif origin in (typing.Type, type):
            return self._subclass_schema(obj)
        elif origin in {typing.Sequence, collections.abc.Sequence}:
            return self._sequence_schema(obj)
        elif origin in {typing.Iterable, collections.abc.Iterable, typing.Generator, collections.abc.Generator}:
            return self._iterable_schema(obj)
        elif origin in (re.Pattern, typing.Pattern):
            return self._pattern_schema(obj)
        if self._arbitrary_types:
            return self._arbitrary_type_schema(origin)
        return self._unknown_type_schema(obj)

    def _generate_td_field_schema(self, name: str, field_info: FieldInfo, decorators: DecoratorInfos, *, required: bool=True) -> core_schema.TypedDictField:
        """Prepare a TypedDictField to represent a model or typeddict field."""
        common_field = self._common_field_schema(name, field_info, decorators)
        return core_schema.typed_dict_field(common_field['schema'], required=False if not field_info.is_required() else required, serialization_exclude=common_field['serialization_exclude'], validation_alias=common_field['validation_alias'], serialization_alias=common_field['serialization_alias'], metadata=common_field['metadata'])

    def _generate_md_field_schema(self, name: str, field_info: FieldInfo, decorators: DecoratorInfos) -> core_schema.ModelField:
        """Prepare a ModelField to represent a model field."""
        common_field = self._common_field_schema(name, field_info, decorators)
        return core_schema.model_field(common_field['schema'], serialization_exclude=common_field['serialization_exclude'], validation_alias=common_field['validation_alias'], serialization_alias=common_field['serialization_alias'], frozen=common_field['frozen'], metadata=common_field['metadata'])

    def _generate_dc_field_schema(self, name: str, field_info: FieldInfo, decorators: DecoratorInfos) -> core_schema.DataclassField:
        """Prepare a DataclassField to represent the parameter/field, of a dataclass."""
        common_field = self._common_field_schema(name, field_info, decorators)
        return core_schema.dataclass_field(name, common_field['schema'], init=field_info.init, init_only=field_info.init_var or None, kw_only=None if field_info.kw_only else False, serialization_exclude=common_field['serialization_exclude'], validation_alias=common_field['validation_alias'], serialization_alias=common_field['serialization_alias'], frozen=common_field['frozen'], metadata=common_field['metadata'])

    @staticmethod
    def _apply_alias_generator_to_field_info(alias_generator: Callable[[str], str] | AliasGenerator, field_info: FieldInfo, field_name: str) -> None:
        """Apply an alias_generator to aliases on a FieldInfo instance if appropriate.

        Args:
            alias_generator: A callable that takes a string and returns a string, or an AliasGenerator instance.
            field_info: The FieldInfo instance to which the alias_generator is (maybe) applied.
            field_name: The name of the field from which to generate the alias.
        """
        if field_info.alias_priority is None or field_info.alias_priority <= 1 or field_info.alias is None or (field_info.validation_alias is None) or (field_info.serialization_alias is None):
            alias, validation_alias, serialization_alias = (None, None, None)
            if isinstance(alias_generator, AliasGenerator):
                alias, validation_alias, serialization_alias = alias_generator.generate_aliases(field_name)
            elif isinstance(alias_generator, Callable):
                alias = alias_generator(field_name)
                if not isinstance(alias, str):
                    raise TypeError(f'alias_generator {alias_generator} must return str, not {alias.__class__}')
            if field_info.alias_priority is None or field_info.alias_priority <= 1:
                field_info.alias_priority = 1
            if field_info.alias_priority == 1:
                field_info.serialization_alias = serialization_alias or alias
                field_info.validation_alias = validation_alias or alias
                field_info.alias = alias
            if field_info.alias is None:
                field_info.alias = alias
            if field_info.serialization_alias is None:
                field_info.serialization_alias = serialization_alias or alias
            if field_info.validation_alias is None:
                field_info.validation_alias = validation_alias or alias

    @staticmethod
    def _apply_alias_generator_to_computed_field_info(alias_generator: Callable[[str], str] | AliasGenerator, computed_field_info: ComputedFieldInfo, computed_field_name: str):
        """Apply an alias_generator to alias on a ComputedFieldInfo instance if appropriate.

        Args:
            alias_generator: A callable that takes a string and returns a string, or an AliasGenerator instance.
            computed_field_info: The ComputedFieldInfo instance to which the alias_generator is (maybe) applied.
            computed_field_name: The name of the computed field from which to generate the alias.
        """
        if computed_field_info.alias_priority is None or computed_field_info.alias_priority <= 1 or computed_field_info.alias is None:
            alias, validation_alias, serialization_alias = (None, None, None)
            if isinstance(alias_generator, AliasGenerator):
                alias, validation_alias, serialization_alias = alias_generator.generate_aliases(computed_field_name)
            elif isinstance(alias_generator, Callable):
                alias = alias_generator(computed_field_name)
                if not isinstance(alias, str):
                    raise TypeError(f'alias_generator {alias_generator} must return str, not {alias.__class__}')
            if computed_field_info.alias_priority is None or computed_field_info.alias_priority <= 1:
                computed_field_info.alias_priority = 1
            if computed_field_info.alias_priority == 1:
                computed_field_info.alias = serialization_alias or alias

    def _common_field_schema(self, name: str, field_info: FieldInfo, decorators: DecoratorInfos) -> _CommonField:
        from .. import AliasChoices, AliasPath
        from ..fields import FieldInfo
        if has_instance_in_type(field_info.annotation, (ForwardRef, str)):
            types_namespace = self._types_namespace
            if self._typevars_map:
                types_namespace = (types_namespace or {}).copy()
                types_namespace.update({k.__name__: v for k, v in self._typevars_map.items()})
            evaluated = _typing_extra.eval_type_lenient(field_info.annotation, types_namespace)
            if evaluated is not field_info.annotation and (not has_instance_in_type(evaluated, PydanticRecursiveRef)):
                new_field_info = FieldInfo.from_annotation(evaluated)
                field_info.annotation = new_field_info.annotation
                for k, v in new_field_info._attributes_set.items():
                    if k not in field_info._attributes_set and k not in field_info.metadata_lookup:
                        setattr(field_info, k, v)
                field_info.metadata = [*new_field_info.metadata, *field_info.metadata]
        source_type, annotations = (field_info.annotation, field_info.metadata)

        def set_discriminator(schema: CoreSchema) -> CoreSchema:
            schema = self._apply_discriminator_to_union(schema, field_info.discriminator)
            return schema
        with self.field_name_stack.push(name):
            if field_info.discriminator is not None:
                schema = self._apply_annotations(source_type, annotations, transform_inner_schema=set_discriminator)
            else:
                schema = self._apply_annotations(source_type, annotations)
        this_field_validators = filter_field_decorator_info_by_field(decorators.validators.values(), name)
        if _validators_require_validate_default(this_field_validators):
            field_info.validate_default = True
        each_item_validators = [v for v in this_field_validators if v.info.each_item is True]
        this_field_validators = [v for v in this_field_validators if v not in each_item_validators]
        schema = apply_each_item_validators(schema, each_item_validators, name)
        schema = apply_validators(schema, filter_field_decorator_info_by_field(this_field_validators, name), name)
        schema = apply_validators(schema, filter_field_decorator_info_by_field(decorators.field_validators.values(), name), name)
        if not field_info.is_required():
            schema = wrap_default(field_info, schema)
        schema = self._apply_field_serializers(schema, filter_field_decorator_info_by_field(decorators.field_serializers.values(), name))
        json_schema_updates = {'title': field_info.title, 'description': field_info.description, 'examples': to_jsonable_python(field_info.examples)}
        json_schema_updates = {k: v for k, v in json_schema_updates.items() if v is not None}
        json_schema_extra = field_info.json_schema_extra
        metadata = build_metadata_dict(js_annotation_functions=[get_json_schema_update_func(json_schema_updates, json_schema_extra)])
        alias_generator = self._config_wrapper.alias_generator
        if alias_generator is not None:
            self._apply_alias_generator_to_field_info(alias_generator, field_info, name)
        if isinstance(field_info.validation_alias, (AliasChoices, AliasPath)):
            validation_alias = field_info.validation_alias.convert_to_aliases()
        else:
            validation_alias = field_info.validation_alias
        return _common_field(schema, serialization_exclude=True if field_info.exclude else None, validation_alias=validation_alias, serialization_alias=field_info.serialization_alias, frozen=field_info.frozen, metadata=metadata)

    def _union_schema(self, union_type: Any) -> core_schema.CoreSchema:
        """Generate schema for a Union."""
        args = self._get_args_resolving_forward_refs(union_type, required=True)
        choices: list[CoreSchema] = []
        nullable = False
        for arg in args:
            if arg is None or arg is _typing_extra.NoneType:
                nullable = True
            else:
                choices.append(self.generate_schema(arg))
        if len(choices) == 1:
            s = choices[0]
        else:
            choices_with_tags: list[CoreSchema | tuple[CoreSchema, str]] = []
            for choice in choices:
                metadata = choice.get('metadata')
                if isinstance(metadata, dict):
                    tag = metadata.get(_core_utils.TAGGED_UNION_TAG_KEY)
                    if tag is not None:
                        choices_with_tags.append((choice, tag))
                    else:
                        choices_with_tags.append(choice)
            s = core_schema.union_schema(choices_with_tags)
        if nullable:
            s = core_schema.nullable_schema(s)
        return s

    def _type_alias_type_schema(self, obj: Any) -> CoreSchema:
        with self.defs.get_schema_or_ref(obj) as (ref, maybe_schema):
            if maybe_schema is not None:
                return maybe_schema
            origin = get_origin(obj) or obj
            annotation = origin.__value__
            typevars_map = get_standard_typevars_map(obj)
            with self._types_namespace_stack.push(origin):
                annotation = _typing_extra.eval_type_lenient(annotation, self._types_namespace)
                annotation = replace_types(annotation, typevars_map)
                schema = self.generate_schema(annotation)
                assert schema['type'] != 'definitions'
                schema['ref'] = ref
            self.defs.definitions[ref] = schema
            return core_schema.definition_reference_schema(ref)

    def _literal_schema(self, literal_type: Any) -> CoreSchema:
        """Generate schema for a Literal."""
        expected = _typing_extra.all_literal_values(literal_type)
        assert expected, f'literal "expected" cannot be empty, obj={literal_type}'
        return core_schema.literal_schema(expected)

    def _typed_dict_schema(self, typed_dict_cls: Any, origin: Any) -> core_schema.CoreSchema:
        """Generate schema for a TypedDict.

        It is not possible to track required/optional keys in TypedDict without __required_keys__
        since TypedDict.__new__ erases the base classes (it replaces them with just `dict`)
        and thus we can track usage of total=True/False
        __required_keys__ was added in Python 3.9
        (https://github.com/miss-islington/cpython/blob/1e9939657dd1f8eb9f596f77c1084d2d351172fc/Doc/library/typing.rst?plain=1#L1546-L1548)
        however it is buggy
        (https://github.com/python/typing_extensions/blob/ac52ac5f2cb0e00e7988bae1e2a1b8257ac88d6d/src/typing_extensions.py#L657-L666).

        On 3.11 but < 3.12 TypedDict does not preserve inheritance information.

        Hence to avoid creating validators that do not do what users expect we only
        support typing.TypedDict on Python >= 3.12 or typing_extension.TypedDict on all versions
        """
        from ..fields import FieldInfo
        with self.defs.get_schema_or_ref(typed_dict_cls) as (typed_dict_ref, maybe_schema):
            if maybe_schema is not None:
                return maybe_schema
            typevars_map = get_standard_typevars_map(typed_dict_cls)
            if origin is not None:
                typed_dict_cls = origin
            if not _SUPPORTS_TYPEDDICT and type(typed_dict_cls).__module__ == 'typing':
                raise PydanticUserError('Please use `typing_extensions.TypedDict` instead of `typing.TypedDict` on Python < 3.12.', code='typed-dict-version')
            try:
                config: ConfigDict | None = get_attribute_from_bases(typed_dict_cls, '__pydantic_config__')
            except AttributeError:
                config = None
            with self._config_wrapper_stack.push(config), self._types_namespace_stack.push(typed_dict_cls):
                core_config = self._config_wrapper.core_config(typed_dict_cls)
                self = self._current_generate_schema
                required_keys: frozenset[str] = typed_dict_cls.__required_keys__
                fields: dict[str, core_schema.TypedDictField] = {}
                decorators = DecoratorInfos.build(typed_dict_cls)
                for field_name, annotation in get_type_hints_infer_globalns(typed_dict_cls, localns=self._types_namespace, include_extras=True).items():
                    annotation = replace_types(annotation, typevars_map)
                    required = field_name in required_keys
                    if get_origin(annotation) == _typing_extra.Required:
                        required = True
                        annotation = self._get_args_resolving_forward_refs(annotation, required=True)[0]
                    elif get_origin(annotation) == _typing_extra.NotRequired:
                        required = False
                        annotation = self._get_args_resolving_forward_refs(annotation, required=True)[0]
                    field_info = FieldInfo.from_annotation(annotation)
                    fields[field_name] = self._generate_td_field_schema(field_name, field_info, decorators, required=required)
                metadata = build_metadata_dict(js_functions=[partial(modify_model_json_schema, cls=typed_dict_cls)], typed_dict_cls=typed_dict_cls)
                td_schema = core_schema.typed_dict_schema(fields, computed_fields=[self._computed_field_schema(d, decorators.field_serializers) for d in decorators.computed_fields.values()], ref=typed_dict_ref, metadata=metadata, config=core_config)
                schema = self._apply_model_serializers(td_schema, decorators.model_serializers.values())
                schema = apply_model_validators(schema, decorators.model_validators.values(), 'all')
                self.defs.definitions[typed_dict_ref] = self._post_process_generated_schema(schema)
                return core_schema.definition_reference_schema(typed_dict_ref)

    def _namedtuple_schema(self, namedtuple_cls: Any, origin: Any) -> core_schema.CoreSchema:
        """Generate schema for a NamedTuple."""
        with self.defs.get_schema_or_ref(namedtuple_cls) as (namedtuple_ref, maybe_schema):
            if maybe_schema is not None:
                return maybe_schema
            typevars_map = get_standard_typevars_map(namedtuple_cls)
            if origin is not None:
                namedtuple_cls = origin
            annotations: dict[str, Any] = get_type_hints_infer_globalns(namedtuple_cls, include_extras=True, localns=self._types_namespace)
            if not annotations:
                annotations = {k: Any for k in namedtuple_cls._fields}
            if typevars_map:
                annotations = {field_name: replace_types(annotation, typevars_map) for field_name, annotation in annotations.items()}
            arguments_schema = core_schema.arguments_schema([self._generate_parameter_schema(field_name, annotation, default=namedtuple_cls._field_defaults.get(field_name, Parameter.empty)) for field_name, annotation in annotations.items()], metadata=build_metadata_dict(js_prefer_positional_arguments=True))
            return core_schema.call_schema(arguments_schema, namedtuple_cls, ref=namedtuple_ref)

    def _generate_parameter_schema(self, name: str, annotation: type[Any], default: Any=Parameter.empty, mode: Literal['positional_only', 'positional_or_keyword', 'keyword_only'] | None=None) -> core_schema.ArgumentsParameter:
        """Prepare a ArgumentsParameter to represent a field in a namedtuple or function signature."""
        from ..fields import FieldInfo
        if default is Parameter.empty:
            field = FieldInfo.from_annotation(annotation)
        else:
            field = FieldInfo.from_annotated_attribute(annotation, default)
        assert field.annotation is not None, 'field.annotation should not be None when generating a schema'
        source_type, annotations = (field.annotation, field.metadata)
        with self.field_name_stack.push(name):
            schema = self._apply_annotations(source_type, annotations)
        if not field.is_required():
            schema = wrap_default(field, schema)
        parameter_schema = core_schema.arguments_parameter(name, schema)
        if mode is not None:
            parameter_schema['mode'] = mode
        if field.alias is not None:
            parameter_schema['alias'] = field.alias
        else:
            alias_generator = self._config_wrapper.alias_generator
            if isinstance(alias_generator, AliasGenerator) and alias_generator.alias is not None:
                parameter_schema['alias'] = alias_generator.alias(name)
            elif isinstance(alias_generator, Callable):
                parameter_schema['alias'] = alias_generator(name)
        return parameter_schema

    def _tuple_schema(self, tuple_type: Any) -> core_schema.CoreSchema:
        """Generate schema for a Tuple, e.g. `tuple[int, str]` or `tuple[int, ...]`."""
        typevars_map = get_standard_typevars_map(tuple_type)
        params = self._get_args_resolving_forward_refs(tuple_type)
        if typevars_map and params:
            params = tuple((replace_types(param, typevars_map) for param in params))
        if not params:
            if tuple_type in TUPLE_TYPES:
                return core_schema.tuple_schema([core_schema.any_schema()], variadic_item_index=0)
            else:
                return core_schema.tuple_schema([])
        elif params[-1] is Ellipsis:
            if len(params) == 2:
                return core_schema.tuple_schema([self.generate_schema(params[0])], variadic_item_index=0)
            else:
                raise ValueError('Variable tuples can only have one type')
        elif len(params) == 1 and params[0] == ():
            return core_schema.tuple_schema([])
        else:
            return core_schema.tuple_schema([self.generate_schema(param) for param in params])

    def _type_schema(self) -> core_schema.CoreSchema:
        return core_schema.custom_error_schema(core_schema.is_instance_schema(type), custom_error_type='is_type', custom_error_message='Input should be a type')

    def _union_is_subclass_schema(self, union_type: Any) -> core_schema.CoreSchema:
        """Generate schema for `Type[Union[X, ...]]`."""
        args = self._get_args_resolving_forward_refs(union_type, required=True)
        return core_schema.union_schema([self.generate_schema(typing.Type[args]) for args in args])

    def _subclass_schema(self, type_: Any) -> core_schema.CoreSchema:
        """Generate schema for a Type, e.g. `Type[int]`."""
        type_param = self._get_first_arg_or_any(type_)
        if type_param == Any:
            return self._type_schema()
        elif isinstance(type_param, typing.TypeVar):
            if type_param.__bound__:
                if _typing_extra.origin_is_union(get_origin(type_param.__bound__)):
                    return self._union_is_subclass_schema(type_param.__bound__)
                return core_schema.is_subclass_schema(type_param.__bound__)
            elif type_param.__constraints__:
                return core_schema.union_schema([self.generate_schema(typing.Type[c]) for c in type_param.__constraints__])
            else:
                return self._type_schema()
        elif _typing_extra.origin_is_union(get_origin(type_param)):
            return self._union_is_subclass_schema(type_param)
        else:
            return core_schema.is_subclass_schema(type_param)

    def _sequence_schema(self, sequence_type: Any) -> core_schema.CoreSchema:
        """Generate schema for a Sequence, e.g. `Sequence[int]`."""
        item_type = self._get_first_arg_or_any(sequence_type)
        list_schema = core_schema.list_schema(self.generate_schema(item_type))
        python_schema = core_schema.is_instance_schema(typing.Sequence, cls_repr='Sequence')
        if item_type != Any:
            from ._validators import sequence_validator
            python_schema = core_schema.chain_schema([python_schema, core_schema.no_info_wrap_validator_function(sequence_validator, list_schema)])
        return core_schema.json_or_python_schema(json_schema=list_schema, python_schema=python_schema)

    def _iterable_schema(self, type_: Any) -> core_schema.GeneratorSchema:
        """Generate a schema for an `Iterable`."""
        item_type = self._get_first_arg_or_any(type_)
        return core_schema.generator_schema(self.generate_schema(item_type))

    def _pattern_schema(self, pattern_type: Any) -> core_schema.CoreSchema:
        from . import _validators
        metadata = build_metadata_dict(js_functions=[lambda _1, _2: {'type': 'string', 'format': 'regex'}])
        ser = core_schema.plain_serializer_function_ser_schema(attrgetter('pattern'), when_used='json', return_schema=core_schema.str_schema())
        if pattern_type == typing.Pattern or pattern_type == re.Pattern:
            return core_schema.no_info_plain_validator_function(_validators.pattern_either_validator, serialization=ser, metadata=metadata)
        param = self._get_args_resolving_forward_refs(pattern_type, required=True)[0]
        if param == str:
            return core_schema.no_info_plain_validator_function(_validators.pattern_str_validator, serialization=ser, metadata=metadata)
        elif param == bytes:
            return core_schema.no_info_plain_validator_function(_validators.pattern_bytes_validator, serialization=ser, metadata=metadata)
        else:
            raise PydanticSchemaGenerationError(f'Unable to generate pydantic-core schema for {pattern_type!r}.')

    def _hashable_schema(self) -> core_schema.CoreSchema:
        return core_schema.custom_error_schema(core_schema.is_instance_schema(collections.abc.Hashable), custom_error_type='is_hashable', custom_error_message='Input should be hashable')

    def _dataclass_schema(self, dataclass: type[StandardDataclass], origin: type[StandardDataclass] | None) -> core_schema.CoreSchema:
        """Generate schema for a dataclass."""
        with self.defs.get_schema_or_ref(dataclass) as (dataclass_ref, maybe_schema):
            if maybe_schema is not None:
                return maybe_schema
            typevars_map = get_standard_typevars_map(dataclass)
            if origin is not None:
                dataclass = origin
            config = getattr(dataclass, '__pydantic_config__', None)
            with self._config_wrapper_stack.push(config), self._types_namespace_stack.push(dataclass):
                core_config = self._config_wrapper.core_config(dataclass)
                self = self._current_generate_schema
                from ..dataclasses import is_pydantic_dataclass
                if is_pydantic_dataclass(dataclass):
                    fields = deepcopy(dataclass.__pydantic_fields__)
                    if typevars_map:
                        for field in fields.values():
                            field.apply_typevars_map(typevars_map, self._types_namespace)
                else:
                    fields = collect_dataclass_fields(dataclass, self._types_namespace, typevars_map=typevars_map)
                if config and config.get('extra') == 'allow':
                    for field_name, field in fields.items():
                        if field.init is False:
                            raise PydanticUserError(f'Field {field_name} has `init=False` and dataclass has config setting `extra="allow"`. This combination is not allowed.', code='dataclass-init-false-extra-allow')
                decorators = dataclass.__dict__.get('__pydantic_decorators__') or DecoratorInfos.build(dataclass)
                args = sorted((self._generate_dc_field_schema(k, v, decorators) for k, v in fields.items()), key=lambda a: a.get('kw_only') is not False)
                has_post_init = hasattr(dataclass, '__post_init__')
                has_slots = hasattr(dataclass, '__slots__')
                args_schema = core_schema.dataclass_args_schema(dataclass.__name__, args, computed_fields=[self._computed_field_schema(d, decorators.field_serializers) for d in decorators.computed_fields.values()], collect_init_only=has_post_init)
                inner_schema = apply_validators(args_schema, decorators.root_validators.values(), None)
                model_validators = decorators.model_validators.values()
                inner_schema = apply_model_validators(inner_schema, model_validators, 'inner')
                dc_schema = core_schema.dataclass_schema(dataclass, inner_schema, post_init=has_post_init, ref=dataclass_ref, fields=[field.name for field in dataclasses.fields(dataclass)], slots=has_slots, config=core_config)
                schema = self._apply_model_serializers(dc_schema, decorators.model_serializers.values())
                schema = apply_model_validators(schema, model_validators, 'outer')
                self.defs.definitions[dataclass_ref] = self._post_process_generated_schema(schema)
                return core_schema.definition_reference_schema(dataclass_ref)

    def _callable_schema(self, function: Callable[..., Any]) -> core_schema.CallSchema:
        """Generate schema for a Callable.

        TODO support functional validators once we support them in Config
        """
        sig = signature(function)
        type_hints = _typing_extra.get_function_type_hints(function)
        mode_lookup: dict[_ParameterKind, Literal['positional_only', 'positional_or_keyword', 'keyword_only']] = {Parameter.POSITIONAL_ONLY: 'positional_only', Parameter.POSITIONAL_OR_KEYWORD: 'positional_or_keyword', Parameter.KEYWORD_ONLY: 'keyword_only'}
        arguments_list: list[core_schema.ArgumentsParameter] = []
        var_args_schema: core_schema.CoreSchema | None = None
        var_kwargs_schema: core_schema.CoreSchema | None = None
        for name, p in sig.parameters.items():
            if p.annotation is sig.empty:
                annotation = Any
            else:
                annotation = type_hints[name]
            parameter_mode = mode_lookup.get(p.kind)
            if parameter_mode is not None:
                arg_schema = self._generate_parameter_schema(name, annotation, p.default, parameter_mode)
                arguments_list.append(arg_schema)
            elif p.kind == Parameter.VAR_POSITIONAL:
                var_args_schema = self.generate_schema(annotation)
            else:
                assert p.kind == Parameter.VAR_KEYWORD, p.kind
                var_kwargs_schema = self.generate_schema(annotation)
        return_schema: core_schema.CoreSchema | None = None
        config_wrapper = self._config_wrapper
        if config_wrapper.validate_return:
            return_hint = type_hints.get('return')
            if return_hint is not None:
                return_schema = self.generate_schema(return_hint)
        return core_schema.call_schema(core_schema.arguments_schema(arguments_list, var_args_schema=var_args_schema, var_kwargs_schema=var_kwargs_schema, populate_by_name=config_wrapper.populate_by_name), function, return_schema=return_schema)

    def _unsubstituted_typevar_schema(self, typevar: typing.TypeVar) -> core_schema.CoreSchema:
        assert isinstance(typevar, typing.TypeVar)
        bound = typevar.__bound__
        constraints = typevar.__constraints__
        default = getattr(typevar, '__default__', None)
        if (bound is not None) + (len(constraints) != 0) + (default is not None) > 1:
            raise NotImplementedError('Pydantic does not support mixing more than one of TypeVar bounds, constraints and defaults')
        if default is not None:
            return self.generate_schema(default)
        elif constraints:
            return self._union_schema(typing.Union[constraints])
        elif bound:
            schema = self.generate_schema(bound)
            schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(lambda x, h: h(x), schema=core_schema.any_schema())
            return schema
        else:
            return core_schema.any_schema()

    def _computed_field_schema(self, d: Decorator[ComputedFieldInfo], field_serializers: dict[str, Decorator[FieldSerializerDecoratorInfo]]) -> core_schema.ComputedField:
        try:
            return_type = _decorators.get_function_return_type(d.func, d.info.return_type, self._types_namespace)
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        if return_type is PydanticUndefined:
            raise PydanticUserError('Computed field is missing return type annotation or specifying `return_type` to the `@computed_field` decorator (e.g. `@computed_field(return_type=int|str)`)', code='model-field-missing-annotation')
        return_type = replace_types(return_type, self._typevars_map)
        d.info = dataclasses.replace(d.info, return_type=return_type)
        return_type_schema = self.generate_schema(return_type)
        return_type_schema = self._apply_field_serializers(return_type_schema, filter_field_decorator_info_by_field(field_serializers.values(), d.cls_var_name), computed_field=True)
        alias_generator = self._config_wrapper.alias_generator
        if alias_generator is not None:
            self._apply_alias_generator_to_computed_field_info(alias_generator=alias_generator, computed_field_info=d.info, computed_field_name=d.cls_var_name)

        def set_computed_field_metadata(schema: CoreSchemaOrField, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
            json_schema = handler(schema)
            json_schema['readOnly'] = True
            title = d.info.title
            if title is not None:
                json_schema['title'] = title
            description = d.info.description
            if description is not None:
                json_schema['description'] = description
            examples = d.info.examples
            if examples is not None:
                json_schema['examples'] = to_jsonable_python(examples)
            json_schema_extra = d.info.json_schema_extra
            if json_schema_extra is not None:
                add_json_schema_extra(json_schema, json_schema_extra)
            return json_schema
        metadata = build_metadata_dict(js_annotation_functions=[set_computed_field_metadata])
        return core_schema.computed_field(d.cls_var_name, return_schema=return_type_schema, alias=d.info.alias, metadata=metadata)

    def _annotated_schema(self, annotated_type: Any) -> core_schema.CoreSchema:
        """Generate schema for an Annotated type, e.g. `Annotated[int, Field(...)]` or `Annotated[int, Gt(0)]`."""
        from ..fields import FieldInfo
        source_type, *annotations = self._get_args_resolving_forward_refs(annotated_type, required=True)
        schema = self._apply_annotations(source_type, annotations)
        for annotation in annotations:
            if isinstance(annotation, FieldInfo):
                schema = wrap_default(annotation, schema)
        return schema

    def _get_prepare_pydantic_annotations_for_known_type(self, obj: Any, annotations: tuple[Any, ...]) -> tuple[Any, list[Any]] | None:
        from ._std_types_schema import PREPARE_METHODS
        try:
            hash(obj)
        except TypeError:
            return None
        for gen in PREPARE_METHODS:
            res = gen(obj, annotations, self._config_wrapper.config_dict)
            if res is not None:
                return res
        return None

    def _apply_annotations(self, source_type: Any, annotations: list[Any], transform_inner_schema: Callable[[CoreSchema], CoreSchema]=lambda x: x) -> CoreSchema:
        """Apply arguments from `Annotated` or from `FieldInfo` to a schema.

        This gets called by `GenerateSchema._annotated_schema` but differs from it in that it does
        not expect `source_type` to be an `Annotated` object, it expects it to be  the first argument of that
        (in other words, `GenerateSchema._annotated_schema` just unpacks `Annotated`, this process it).
        """
        annotations = list(_known_annotated_metadata.expand_grouped_metadata(annotations))
        res = self._get_prepare_pydantic_annotations_for_known_type(source_type, tuple(annotations))
        if res is not None:
            source_type, annotations = res
        pydantic_js_annotation_functions: list[GetJsonSchemaFunction] = []

        def inner_handler(obj: Any) -> CoreSchema:
            from_property = self._generate_schema_from_property(obj, obj)
            if from_property is None:
                schema = self._generate_schema(obj)
            else:
                schema = from_property
            metadata_js_function = _extract_get_pydantic_json_schema(obj, schema)
            if metadata_js_function is not None:
                metadata_schema = resolve_original_schema(schema, self.defs.definitions)
                if metadata_schema is not None:
                    self._add_js_function(metadata_schema, metadata_js_function)
            return transform_inner_schema(schema)
        get_inner_schema = CallbackGetCoreSchemaHandler(inner_handler, self)
        for annotation in annotations:
            if annotation is None:
                continue
            get_inner_schema = self._get_wrapped_inner_schema(get_inner_schema, annotation, pydantic_js_annotation_functions)
        schema = get_inner_schema(source_type)
        if pydantic_js_annotation_functions:
            metadata = CoreMetadataHandler(schema).metadata
            metadata.setdefault('pydantic_js_annotation_functions', []).extend(pydantic_js_annotation_functions)
        return _add_custom_serialization_from_json_encoders(self._config_wrapper.json_encoders, source_type, schema)

    def _apply_single_annotation(self, schema: core_schema.CoreSchema, metadata: Any) -> core_schema.CoreSchema:
        from ..fields import FieldInfo
        if isinstance(metadata, FieldInfo):
            for field_metadata in metadata.metadata:
                schema = self._apply_single_annotation(schema, field_metadata)
            if metadata.discriminator is not None:
                schema = self._apply_discriminator_to_union(schema, metadata.discriminator)
            return schema
        if schema['type'] == 'nullable':
            inner = schema.get('schema', core_schema.any_schema())
            inner = self._apply_single_annotation(inner, metadata)
            if inner:
                schema['schema'] = inner
            return schema
        original_schema = schema
        ref = schema.get('ref', None)
        if ref is not None:
            schema = schema.copy()
            new_ref = ref + f'_{repr(metadata)}'
            if new_ref in self.defs.definitions:
                return self.defs.definitions[new_ref]
            schema['ref'] = new_ref
        elif schema['type'] == 'definition-ref':
            ref = schema['schema_ref']
            if ref in self.defs.definitions:
                schema = self.defs.definitions[ref].copy()
                new_ref = ref + f'_{repr(metadata)}'
                if new_ref in self.defs.definitions:
                    return self.defs.definitions[new_ref]
                schema['ref'] = new_ref
        maybe_updated_schema = _known_annotated_metadata.apply_known_metadata(metadata, schema.copy())
        if maybe_updated_schema is not None:
            return maybe_updated_schema
        return original_schema

    def _apply_single_annotation_json_schema(self, schema: core_schema.CoreSchema, metadata: Any) -> core_schema.CoreSchema:
        from ..fields import FieldInfo
        if isinstance(metadata, FieldInfo):
            for field_metadata in metadata.metadata:
                schema = self._apply_single_annotation_json_schema(schema, field_metadata)
            json_schema_update: JsonSchemaValue = {}
            if metadata.title:
                json_schema_update['title'] = metadata.title
            if metadata.description:
                json_schema_update['description'] = metadata.description
            if metadata.examples:
                json_schema_update['examples'] = to_jsonable_python(metadata.examples)
            json_schema_extra = metadata.json_schema_extra
            if json_schema_update or json_schema_extra:
                CoreMetadataHandler(schema).metadata.setdefault('pydantic_js_annotation_functions', []).append(get_json_schema_update_func(json_schema_update, json_schema_extra))
        return schema

    def _get_wrapped_inner_schema(self, get_inner_schema: GetCoreSchemaHandler, annotation: Any, pydantic_js_annotation_functions: list[GetJsonSchemaFunction]) -> CallbackGetCoreSchemaHandler:
        metadata_get_schema: GetCoreSchemaFunction = getattr(annotation, '__get_pydantic_core_schema__', None) or (lambda source, handler: handler(source))

        def new_handler(source: Any) -> core_schema.CoreSchema:
            schema = metadata_get_schema(source, get_inner_schema)
            schema = self._apply_single_annotation(schema, annotation)
            schema = self._apply_single_annotation_json_schema(schema, annotation)
            metadata_js_function = _extract_get_pydantic_json_schema(annotation, schema)
            if metadata_js_function is not None:
                pydantic_js_annotation_functions.append(metadata_js_function)
            return schema
        return CallbackGetCoreSchemaHandler(new_handler, self)

    def _apply_field_serializers(self, schema: core_schema.CoreSchema, serializers: list[Decorator[FieldSerializerDecoratorInfo]], computed_field: bool=False) -> core_schema.CoreSchema:
        """Apply field serializers to a schema."""
        if serializers:
            schema = copy(schema)
            if schema['type'] == 'definitions':
                inner_schema = schema['schema']
                schema['schema'] = self._apply_field_serializers(inner_schema, serializers)
                return schema
            else:
                ref = typing.cast('str|None', schema.get('ref', None))
                if ref is not None:
                    schema = core_schema.definition_reference_schema(ref)
            serializer = serializers[-1]
            is_field_serializer, info_arg = inspect_field_serializer(serializer.func, serializer.info.mode, computed_field=computed_field)
            try:
                return_type = _decorators.get_function_return_type(serializer.func, serializer.info.return_type, self._types_namespace)
            except NameError as e:
                raise PydanticUndefinedAnnotation.from_name_error(e) from e
            if return_type is PydanticUndefined:
                return_schema = None
            else:
                return_schema = self.generate_schema(return_type)
            if serializer.info.mode == 'wrap':
                schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(serializer.func, is_field_serializer=is_field_serializer, info_arg=info_arg, return_schema=return_schema, when_used=serializer.info.when_used)
            else:
                assert serializer.info.mode == 'plain'
                schema['serialization'] = core_schema.plain_serializer_function_ser_schema(serializer.func, is_field_serializer=is_field_serializer, info_arg=info_arg, return_schema=return_schema, when_used=serializer.info.when_used)
        return schema

    def _apply_model_serializers(self, schema: core_schema.CoreSchema, serializers: Iterable[Decorator[ModelSerializerDecoratorInfo]]) -> core_schema.CoreSchema:
        """Apply model serializers to a schema."""
        ref: str | None = schema.pop('ref', None)
        if serializers:
            serializer = list(serializers)[-1]
            info_arg = inspect_model_serializer(serializer.func, serializer.info.mode)
            try:
                return_type = _decorators.get_function_return_type(serializer.func, serializer.info.return_type, self._types_namespace)
            except NameError as e:
                raise PydanticUndefinedAnnotation.from_name_error(e) from e
            if return_type is PydanticUndefined:
                return_schema = None
            else:
                return_schema = self.generate_schema(return_type)
            if serializer.info.mode == 'wrap':
                ser_schema: core_schema.SerSchema = core_schema.wrap_serializer_function_ser_schema(serializer.func, info_arg=info_arg, return_schema=return_schema, when_used=serializer.info.when_used)
            else:
                ser_schema = core_schema.plain_serializer_function_ser_schema(serializer.func, info_arg=info_arg, return_schema=return_schema, when_used=serializer.info.when_used)
            schema['serialization'] = ser_schema
        if ref:
            schema['ref'] = ref
        return schema