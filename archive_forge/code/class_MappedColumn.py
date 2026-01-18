from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import strategy_options
from .base import _DeclarativeMapped
from .base import class_mapper
from .descriptor_props import CompositeProperty
from .descriptor_props import ConcreteInheritedProperty
from .descriptor_props import SynonymProperty
from .interfaces import _AttributeOptions
from .interfaces import _DEFAULT_ATTRIBUTE_OPTIONS
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .interfaces import PropComparator
from .interfaces import StrategizedProperty
from .relationships import RelationshipProperty
from .util import de_stringify_annotation
from .util import de_stringify_union_elements
from .. import exc as sa_exc
from .. import ForeignKey
from .. import log
from .. import util
from ..sql import coercions
from ..sql import roles
from ..sql.base import _NoArg
from ..sql.schema import Column
from ..sql.schema import SchemaConst
from ..sql.type_api import TypeEngine
from ..util.typing import de_optionalize_union_types
from ..util.typing import is_fwd_ref
from ..util.typing import is_optional_union
from ..util.typing import is_pep593
from ..util.typing import is_pep695
from ..util.typing import is_union
from ..util.typing import Self
from ..util.typing import typing_get_args
class MappedColumn(_IntrospectsAnnotations, _MapsColumns[_T], _DeclarativeMapped[_T]):
    """Maps a single :class:`_schema.Column` on a class.

    :class:`_orm.MappedColumn` is a specialization of the
    :class:`_orm.ColumnProperty` class and is oriented towards declarative
    configuration.

    To construct :class:`_orm.MappedColumn` objects, use the
    :func:`_orm.mapped_column` constructor function.

    .. versionadded:: 2.0


    """
    __slots__ = ('column', '_creation_order', '_sort_order', 'foreign_keys', '_has_nullable', '_has_insert_default', 'deferred', 'deferred_group', 'deferred_raiseload', 'active_history', '_attribute_options', '_has_dataclass_arguments', '_use_existing_column')
    deferred: Union[_NoArg, bool]
    deferred_raiseload: bool
    deferred_group: Optional[str]
    column: Column[_T]
    foreign_keys: Optional[Set[ForeignKey]]
    _attribute_options: _AttributeOptions

    def __init__(self, *arg: Any, **kw: Any):
        self._attribute_options = attr_opts = kw.pop('attribute_options', _DEFAULT_ATTRIBUTE_OPTIONS)
        self._use_existing_column = kw.pop('use_existing_column', False)
        self._has_dataclass_arguments = attr_opts is not None and attr_opts != _DEFAULT_ATTRIBUTE_OPTIONS and any((attr_opts[i] is not _NoArg.NO_ARG for i, attr in enumerate(attr_opts._fields) if attr != 'dataclasses_default'))
        insert_default = kw.pop('insert_default', _NoArg.NO_ARG)
        self._has_insert_default = insert_default is not _NoArg.NO_ARG
        if self._has_insert_default:
            kw['default'] = insert_default
        elif attr_opts.dataclasses_default is not _NoArg.NO_ARG:
            kw['default'] = attr_opts.dataclasses_default
        self.deferred_group = kw.pop('deferred_group', None)
        self.deferred_raiseload = kw.pop('deferred_raiseload', None)
        self.deferred = kw.pop('deferred', _NoArg.NO_ARG)
        self.active_history = kw.pop('active_history', False)
        self._sort_order = kw.pop('sort_order', _NoArg.NO_ARG)
        self.column = cast('Column[_T]', Column(*arg, **kw))
        self.foreign_keys = self.column.foreign_keys
        self._has_nullable = 'nullable' in kw and kw.get('nullable') not in (None, SchemaConst.NULL_UNSPECIFIED)
        util.set_creation_order(self)

    def _copy(self, **kw: Any) -> Self:
        new = self.__class__.__new__(self.__class__)
        new.column = self.column._copy(**kw)
        new.deferred = self.deferred
        new.deferred_group = self.deferred_group
        new.deferred_raiseload = self.deferred_raiseload
        new.foreign_keys = new.column.foreign_keys
        new.active_history = self.active_history
        new._has_nullable = self._has_nullable
        new._attribute_options = self._attribute_options
        new._has_insert_default = self._has_insert_default
        new._has_dataclass_arguments = self._has_dataclass_arguments
        new._use_existing_column = self._use_existing_column
        new._sort_order = self._sort_order
        util.set_creation_order(new)
        return new

    @property
    def name(self) -> str:
        return self.column.name

    @property
    def mapper_property_to_assign(self) -> Optional[MapperProperty[_T]]:
        effective_deferred = self.deferred
        if effective_deferred is _NoArg.NO_ARG:
            effective_deferred = bool(self.deferred_group or self.deferred_raiseload)
        if effective_deferred or self.active_history:
            return ColumnProperty(self.column, deferred=effective_deferred, group=self.deferred_group, raiseload=self.deferred_raiseload, attribute_options=self._attribute_options, active_history=self.active_history)
        else:
            return None

    @property
    def columns_to_assign(self) -> List[Tuple[Column[Any], int]]:
        return [(self.column, self._sort_order if self._sort_order is not _NoArg.NO_ARG else 0)]

    def __clause_element__(self) -> Column[_T]:
        return self.column

    def operate(self, op: OperatorType, *other: Any, **kwargs: Any) -> ColumnElement[Any]:
        return op(self.__clause_element__(), *other, **kwargs)

    def reverse_operate(self, op: OperatorType, other: Any, **kwargs: Any) -> ColumnElement[Any]:
        col = self.__clause_element__()
        return op(col._bind_param(op, other), col, **kwargs)

    def found_in_pep593_annotated(self) -> Any:
        return MappedColumn()

    def declarative_scan(self, decl_scan: _ClassScanMapperConfig, registry: _RegistryType, cls: Type[Any], originating_module: Optional[str], key: str, mapped_container: Optional[Type[Mapped[Any]]], annotation: Optional[_AnnotationScanType], extracted_mapped_annotation: Optional[_AnnotationScanType], is_dataclass_field: bool) -> None:
        column = self.column
        if self._use_existing_column and decl_scan.inherits and decl_scan.single:
            if decl_scan.is_deferred:
                raise sa_exc.ArgumentError("Can't use use_existing_column with deferred mappers")
            supercls_mapper = class_mapper(decl_scan.inherits, False)
            colname = column.name if column.name is not None else key
            column = self.column = supercls_mapper.local_table.c.get(colname, column)
        if column.key is None:
            column.key = key
        if column.name is None:
            column.name = key
        sqltype = column.type
        if extracted_mapped_annotation is None:
            if sqltype._isnull and (not self.column.foreign_keys):
                self._raise_for_required(key, cls)
            else:
                return
        self._init_column_for_annotation(cls, registry, extracted_mapped_annotation, originating_module)

    @util.preload_module('sqlalchemy.orm.decl_base')
    def declarative_scan_for_composite(self, registry: _RegistryType, cls: Type[Any], originating_module: Optional[str], key: str, param_name: str, param_annotation: _AnnotationScanType) -> None:
        decl_base = util.preloaded.orm_decl_base
        decl_base._undefer_column_name(param_name, self.column)
        self._init_column_for_annotation(cls, registry, param_annotation, originating_module)

    def _init_column_for_annotation(self, cls: Type[Any], registry: _RegistryType, argument: _AnnotationScanType, originating_module: Optional[str]) -> None:
        sqltype = self.column.type
        if isinstance(argument, str) or is_fwd_ref(argument, check_generic=True):
            assert originating_module is not None
            argument = de_stringify_annotation(cls, argument, originating_module, include_generic=True)
        if is_union(argument):
            assert originating_module is not None
            argument = de_stringify_union_elements(cls, argument, originating_module)
        nullable = is_optional_union(argument)
        if not self._has_nullable:
            self.column.nullable = nullable
        our_type = de_optionalize_union_types(argument)
        use_args_from = None
        our_original_type = our_type
        if is_pep695(our_type):
            our_type = our_type.__value__
        if is_pep593(our_type):
            our_type_is_pep593 = True
            pep_593_components = typing_get_args(our_type)
            raw_pep_593_type = pep_593_components[0]
            if is_optional_union(raw_pep_593_type):
                raw_pep_593_type = de_optionalize_union_types(raw_pep_593_type)
                nullable = True
                if not self._has_nullable:
                    self.column.nullable = nullable
            for elem in pep_593_components[1:]:
                if isinstance(elem, MappedColumn):
                    use_args_from = elem
                    break
        else:
            our_type_is_pep593 = False
            raw_pep_593_type = None
        if use_args_from is not None:
            if not self._has_insert_default and use_args_from.column.default is not None:
                self.column.default = None
            use_args_from.column._merge(self.column)
            sqltype = self.column.type
            if use_args_from.deferred is not _NoArg.NO_ARG and self.deferred is _NoArg.NO_ARG:
                self.deferred = use_args_from.deferred
            if use_args_from.deferred_group is not None and self.deferred_group is None:
                self.deferred_group = use_args_from.deferred_group
            if use_args_from.deferred_raiseload is not None and self.deferred_raiseload is None:
                self.deferred_raiseload = use_args_from.deferred_raiseload
            if use_args_from._use_existing_column and (not self._use_existing_column):
                self._use_existing_column = True
            if use_args_from.active_history:
                self.active_history = use_args_from.active_history
            if use_args_from._sort_order is not None and self._sort_order is _NoArg.NO_ARG:
                self._sort_order = use_args_from._sort_order
            if use_args_from.column.key is not None or use_args_from.column.name is not None:
                util.warn_deprecated("Can't use the 'key' or 'name' arguments in Annotated with mapped_column(); this will be ignored", '2.0.22')
            if use_args_from._has_dataclass_arguments:
                for idx, arg in enumerate(use_args_from._attribute_options._fields):
                    if use_args_from._attribute_options[idx] is not _NoArg.NO_ARG:
                        arg = arg.replace('dataclasses_', '')
                        util.warn_deprecated(f"Argument '{arg}' is a dataclass argument and cannot be specified within a mapped_column() bundled inside of an Annotated object", '2.0.22')
        if sqltype._isnull and (not self.column.foreign_keys):
            new_sqltype = None
            if our_type_is_pep593:
                checks = [our_original_type, raw_pep_593_type]
            else:
                checks = [our_original_type]
            for check_type in checks:
                new_sqltype = registry._resolve_type(check_type)
                if new_sqltype is not None:
                    break
            else:
                if isinstance(our_type, TypeEngine) or (isinstance(our_type, type) and issubclass(our_type, TypeEngine)):
                    raise sa_exc.ArgumentError(f'The type provided inside the {self.column.key!r} attribute Mapped annotation is the SQLAlchemy type {our_type}. Expected a Python type instead')
                else:
                    raise sa_exc.ArgumentError(f'Could not locate SQLAlchemy Core type for Python type {our_type} inside the {self.column.key!r} attribute Mapped annotation')
            self.column._set_type(new_sqltype)