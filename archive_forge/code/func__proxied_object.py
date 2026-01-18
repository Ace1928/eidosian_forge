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
@util.memoized_property
def _proxied_object(self) -> Union[MapperProperty[_T], SQLORMOperations[_T]]:
    attr = getattr(self.parent.class_, self.name)
    if not hasattr(attr, 'property') or not isinstance(attr.property, MapperProperty):
        if isinstance(attr, attributes.QueryableAttribute):
            return attr.comparator
        elif isinstance(attr, SQLORMOperations):
            return attr
        raise sa_exc.InvalidRequestError('synonym() attribute "%s.%s" only supports ORM mapped attributes, got %r' % (self.parent.class_.__name__, self.name, attr))
    return attr.property