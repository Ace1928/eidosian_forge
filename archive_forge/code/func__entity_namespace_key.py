from __future__ import annotations
import collections
from enum import Enum
import itertools
from itertools import zip_longest
import operator
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import roles
from . import visitors
from .cache_key import HasCacheKey  # noqa
from .cache_key import MemoizedHasCacheKey  # noqa
from .traversals import HasCopyInternals  # noqa
from .visitors import ClauseVisitor
from .visitors import ExtendedInternalTraversal
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import util
from ..util import HasMemoized as HasMemoized
from ..util import hybridmethod
from ..util import typing as compat_typing
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeGuard
def _entity_namespace_key(entity: Union[_HasEntityNamespace, ExternallyTraversible], key: str, default: Union[SQLCoreOperations[Any], _NoArg]=NO_ARG) -> SQLCoreOperations[Any]:
    """Return an entry from an entity_namespace.


    Raises :class:`_exc.InvalidRequestError` rather than attribute error
    on not found.

    """
    try:
        ns = _entity_namespace(entity)
        if default is not NO_ARG:
            return getattr(ns, key, default)
        else:
            return getattr(ns, key)
    except AttributeError as err:
        raise exc.InvalidRequestError('Entity namespace for "%s" has no property "%s"' % (entity, key)) from err