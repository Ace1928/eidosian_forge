from __future__ import annotations
from dataclasses import is_dataclass
import inspect
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
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
from . import util as orm_util
from .base import _DeclarativeMapped
from .base import LoaderCallableStatus
from .base import Mapped
from .base import PassiveFlag
from .base import SQLORMOperations
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .interfaces import PropComparator
from .util import _none_set
from .util import de_stringify_annotation
from .. import event
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import util
from ..sql import expression
from ..sql import operators
from ..sql.elements import BindParameter
from ..util.typing import is_fwd_ref
from ..util.typing import is_pep593
from ..util.typing import typing_get_args
@util.preload_module('sqlalchemy.orm.properties')
@util.preload_module('sqlalchemy.orm.decl_base')
def _setup_for_dataclass(self, registry: _RegistryType, cls: Type[Any], originating_module: Optional[str], key: str) -> None:
    MappedColumn = util.preloaded.orm_properties.MappedColumn
    decl_base = util.preloaded.orm_decl_base
    insp = inspect.signature(self.composite_class)
    for param, attr in itertools.zip_longest(insp.parameters.values(), self.attrs):
        if param is None:
            raise sa_exc.ArgumentError(f'number of composite attributes {len(self.attrs)} exceeds that of the number of attributes in class {self.composite_class.__name__} {len(insp.parameters)}')
        if attr is None:
            attr = MappedColumn()
            self.attrs += (attr,)
        if isinstance(attr, MappedColumn):
            attr.declarative_scan_for_composite(registry, cls, originating_module, key, param.name, param.annotation)
        elif isinstance(attr, schema.Column):
            decl_base._undefer_column_name(param.name, attr)