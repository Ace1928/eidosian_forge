from __future__ import annotations
import collections
import dataclasses
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import clsregistry
from . import exc as orm_exc
from . import instrumentation
from . import mapperlib
from ._typing import _O
from ._typing import attr_is_internal_proxy
from .attributes import InstrumentedAttribute
from .attributes import QueryableAttribute
from .base import _is_mapped_class
from .base import InspectionAttr
from .descriptor_props import CompositeProperty
from .descriptor_props import SynonymProperty
from .interfaces import _AttributeOptions
from .interfaces import _DCAttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MappedAttribute
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .mapper import Mapper
from .properties import ColumnProperty
from .properties import MappedColumn
from .util import _extract_mapped_subtype
from .util import _is_mapped_annotation
from .util import class_mapper
from .util import de_stringify_annotation
from .. import event
from .. import exc
from .. import util
from ..sql import expression
from ..sql.base import _NoArg
from ..sql.schema import Column
from ..sql.schema import Table
from ..util import topological
from ..util.typing import _AnnotationScanType
from ..util.typing import is_fwd_ref
from ..util.typing import is_literal
from ..util.typing import Protocol
from ..util.typing import TypedDict
from ..util.typing import typing_get_args
def _prepare_mapper_arguments(self, mapper_kw: _MapperKwArgs) -> None:
    properties = self.properties
    if self.mapper_args_fn:
        mapper_args = self.mapper_args_fn()
    else:
        mapper_args = {}
    if mapper_kw:
        mapper_args.update(mapper_kw)
    if 'properties' in mapper_args:
        properties = dict(properties)
        properties.update(mapper_args['properties'])
    for k in ('version_id_col', 'polymorphic_on'):
        if k in mapper_args:
            v = mapper_args[k]
            mapper_args[k] = self.column_copies.get(v, v)
    if 'primary_key' in mapper_args:
        mapper_args['primary_key'] = [self.column_copies.get(v, v) for v in util.to_list(mapper_args['primary_key'])]
    if 'inherits' in mapper_args:
        inherits_arg = mapper_args['inherits']
        if isinstance(inherits_arg, Mapper):
            inherits_arg = inherits_arg.class_
        if inherits_arg is not self.inherits:
            raise exc.InvalidRequestError('mapper inherits argument given for non-inheriting class %s' % mapper_args['inherits'])
    if self.inherits:
        mapper_args['inherits'] = self.inherits
    if self.inherits and (not mapper_args.get('concrete', False)):
        inherited_mapper = class_mapper(self.inherits, False)
        inherited_table = inherited_mapper.local_table
        if 'exclude_properties' not in mapper_args:
            mapper_args['exclude_properties'] = exclude_properties = {c.key for c in inherited_table.c if c not in inherited_mapper._columntoproperty}.union(inherited_mapper.exclude_properties or ())
            exclude_properties.difference_update([c.key for c in self.declared_columns])
        for k, col in list(properties.items()):
            if not isinstance(col, expression.ColumnElement):
                continue
            if k in inherited_mapper._props:
                p = inherited_mapper._props[k]
                if isinstance(p, ColumnProperty):
                    properties[k] = [col] + p.columns
    result_mapper_args = mapper_args.copy()
    result_mapper_args['properties'] = properties
    self.mapper_args = result_mapper_args