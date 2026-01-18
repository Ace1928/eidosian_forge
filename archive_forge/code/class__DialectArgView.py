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
class _DialectArgView(MutableMapping[str, Any]):
    """A dictionary view of dialect-level arguments in the form
    <dialectname>_<argument_name>.

    """

    def __init__(self, obj):
        self.obj = obj

    def _key(self, key):
        try:
            dialect, value_key = key.split('_', 1)
        except ValueError as err:
            raise KeyError(key) from err
        else:
            return (dialect, value_key)

    def __getitem__(self, key):
        dialect, value_key = self._key(key)
        try:
            opt = self.obj.dialect_options[dialect]
        except exc.NoSuchModuleError as err:
            raise KeyError(key) from err
        else:
            return opt[value_key]

    def __setitem__(self, key, value):
        try:
            dialect, value_key = self._key(key)
        except KeyError as err:
            raise exc.ArgumentError('Keys must be of the form <dialectname>_<argname>') from err
        else:
            self.obj.dialect_options[dialect][value_key] = value

    def __delitem__(self, key):
        dialect, value_key = self._key(key)
        del self.obj.dialect_options[dialect][value_key]

    def __len__(self):
        return sum((len(args._non_defaults) for args in self.obj.dialect_options.values()))

    def __iter__(self):
        return ('%s_%s' % (dialect_name, value_name) for dialect_name in self.obj.dialect_options for value_name in self.obj.dialect_options[dialect_name]._non_defaults)