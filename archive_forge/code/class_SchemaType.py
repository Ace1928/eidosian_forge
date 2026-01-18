from __future__ import annotations
import collections.abc as collections_abc
import datetime as dt
import decimal
import enum
import json
import pickle
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from uuid import UUID as _python_UUID
from . import coercions
from . import elements
from . import operators
from . import roles
from . import type_api
from .base import _NONE_NAME
from .base import NO_ARG
from .base import SchemaEventTarget
from .cache_key import HasCacheKey
from .elements import quoted_name
from .elements import Slice
from .elements import TypeCoerce as type_coerce  # noqa
from .type_api import Emulated
from .type_api import NativeForEmulated  # noqa
from .type_api import to_instance as to_instance
from .type_api import TypeDecorator as TypeDecorator
from .type_api import TypeEngine as TypeEngine
from .type_api import TypeEngineMixin
from .type_api import Variant  # noqa
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..engine import processors
from ..util import langhelpers
from ..util import OrderedDict
from ..util.typing import is_literal
from ..util.typing import Literal
from ..util.typing import typing_get_args
class SchemaType(SchemaEventTarget, TypeEngineMixin):
    """Add capabilities to a type which allow for schema-level DDL to be
    associated with a type.

    Supports types that must be explicitly created/dropped (i.e. PG ENUM type)
    as well as types that are complimented by table or schema level
    constraints, triggers, and other rules.

    :class:`.SchemaType` classes can also be targets for the
    :meth:`.DDLEvents.before_parent_attach` and
    :meth:`.DDLEvents.after_parent_attach` events, where the events fire off
    surrounding the association of the type object with a parent
    :class:`_schema.Column`.

    .. seealso::

        :class:`.Enum`

        :class:`.Boolean`


    """
    _use_schema_map = True
    name: Optional[str]

    def __init__(self, name: Optional[str]=None, schema: Optional[str]=None, metadata: Optional[MetaData]=None, inherit_schema: bool=False, quote: Optional[bool]=None, _create_events: bool=True, _adapted_from: Optional[SchemaType]=None):
        if name is not None:
            self.name = quoted_name(name, quote)
        else:
            self.name = None
        self.schema = schema
        self.metadata = metadata
        self.inherit_schema = inherit_schema
        self._create_events = _create_events
        if _create_events and self.metadata:
            event.listen(self.metadata, 'before_create', util.portable_instancemethod(self._on_metadata_create))
            event.listen(self.metadata, 'after_drop', util.portable_instancemethod(self._on_metadata_drop))
        if _adapted_from:
            self.dispatch = self.dispatch._join(_adapted_from.dispatch)

    def _set_parent(self, column, **kw):
        column._on_table_attach(util.portable_instancemethod(self._set_table))

    def _variant_mapping_for_set_table(self, column):
        if column.type._variant_mapping:
            variant_mapping = dict(column.type._variant_mapping)
            variant_mapping['_default'] = column.type
        else:
            variant_mapping = None
        return variant_mapping

    def _set_table(self, column, table):
        if self.inherit_schema:
            self.schema = table.schema
        elif self.metadata and self.schema is None and self.metadata.schema:
            self.schema = self.metadata.schema
        if not self._create_events:
            return
        variant_mapping = self._variant_mapping_for_set_table(column)
        event.listen(table, 'before_create', util.portable_instancemethod(self._on_table_create, {'variant_mapping': variant_mapping}))
        event.listen(table, 'after_drop', util.portable_instancemethod(self._on_table_drop, {'variant_mapping': variant_mapping}))
        if self.metadata is None:
            event.listen(table.metadata, 'before_create', util.portable_instancemethod(self._on_metadata_create, {'variant_mapping': variant_mapping}))
            event.listen(table.metadata, 'after_drop', util.portable_instancemethod(self._on_metadata_drop, {'variant_mapping': variant_mapping}))

    def copy(self, **kw):
        return self.adapt(cast('Type[TypeEngine[Any]]', self.__class__), _create_events=True)

    @overload
    def adapt(self, cls: Type[_TE], **kw: Any) -> _TE:
        ...

    @overload
    def adapt(self, cls: Type[TypeEngineMixin], **kw: Any) -> TypeEngine[Any]:
        ...

    def adapt(self, cls: Type[Union[TypeEngine[Any], TypeEngineMixin]], **kw: Any) -> TypeEngine[Any]:
        kw.setdefault('_create_events', False)
        kw.setdefault('_adapted_from', self)
        return super().adapt(cls, **kw)

    def create(self, bind, checkfirst=False):
        """Issue CREATE DDL for this type, if applicable."""
        t = self.dialect_impl(bind.dialect)
        if isinstance(t, SchemaType) and t.__class__ is not self.__class__:
            t.create(bind, checkfirst=checkfirst)

    def drop(self, bind, checkfirst=False):
        """Issue DROP DDL for this type, if applicable."""
        t = self.dialect_impl(bind.dialect)
        if isinstance(t, SchemaType) and t.__class__ is not self.__class__:
            t.drop(bind, checkfirst=checkfirst)

    def _on_table_create(self, target, bind, **kw):
        if not self._is_impl_for_variant(bind.dialect, kw):
            return
        t = self.dialect_impl(bind.dialect)
        if isinstance(t, SchemaType) and t.__class__ is not self.__class__:
            t._on_table_create(target, bind, **kw)

    def _on_table_drop(self, target, bind, **kw):
        if not self._is_impl_for_variant(bind.dialect, kw):
            return
        t = self.dialect_impl(bind.dialect)
        if isinstance(t, SchemaType) and t.__class__ is not self.__class__:
            t._on_table_drop(target, bind, **kw)

    def _on_metadata_create(self, target, bind, **kw):
        if not self._is_impl_for_variant(bind.dialect, kw):
            return
        t = self.dialect_impl(bind.dialect)
        if isinstance(t, SchemaType) and t.__class__ is not self.__class__:
            t._on_metadata_create(target, bind, **kw)

    def _on_metadata_drop(self, target, bind, **kw):
        if not self._is_impl_for_variant(bind.dialect, kw):
            return
        t = self.dialect_impl(bind.dialect)
        if isinstance(t, SchemaType) and t.__class__ is not self.__class__:
            t._on_metadata_drop(target, bind, **kw)

    def _is_impl_for_variant(self, dialect, kw):
        variant_mapping = kw.pop('variant_mapping', None)
        if not variant_mapping:
            return True

        def _we_are_the_impl(typ):
            return typ is self or (isinstance(typ, ARRAY) and typ.item_type is self)
        if dialect.name in variant_mapping and _we_are_the_impl(variant_mapping[dialect.name]):
            return True
        elif dialect.name not in variant_mapping:
            return _we_are_the_impl(variant_mapping['_default'])