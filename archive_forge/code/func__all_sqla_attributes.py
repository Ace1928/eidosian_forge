from __future__ import annotations
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import base
from . import collections
from . import exc
from . import interfaces
from . import state
from ._typing import _O
from .attributes import _is_collection_attribute_impl
from .. import util
from ..event import EventTarget
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
def _all_sqla_attributes(self, exclude=None):
    """return an iterator of all classbound attributes that are
        implement :class:`.InspectionAttr`.

        This includes :class:`.QueryableAttribute` as well as extension
        types such as :class:`.hybrid_property` and
        :class:`.AssociationProxy`.

        """
    found: Dict[str, Any] = {}
    for supercls in self.class_.__mro__[0:-1]:
        inherits = supercls.__mro__[1]
        for key in supercls.__dict__:
            found.setdefault(key, supercls)
            if key in inherits.__dict__:
                continue
            val = found[key].__dict__[key]
            if isinstance(val, interfaces.InspectionAttr) and val.is_attribute:
                yield (key, val)