from __future__ import annotations
import contextlib
import datetime
from functools import partial
from functools import wraps
import json
import logging
from numbers import Number
import threading
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from decorator import decorate
from . import exception
from .api import BackendArguments
from .api import BackendFormatted
from .api import CachedValue
from .api import CacheMutex
from .api import CacheReturnType
from .api import CantDeserializeException
from .api import KeyType
from .api import MetaDataType
from .api import NO_VALUE
from .api import SerializedReturnType
from .api import Serializer
from .api import ValuePayload
from .backends import _backend_loader
from .backends import register_backend  # noqa
from .proxy import ProxyBackend
from .util import function_key_generator
from .util import function_multi_key_generator
from .util import repr_obj
from .. import Lock
from .. import NeedRegenerationException
from ..util import coerce_string_conf
from ..util import memoized_property
from ..util import NameRegistry
from ..util import PluginLoader
from ..util.typing import Self
class RegionInvalidationStrategy:
    """Region invalidation strategy interface

    Implement this interface and pass implementation instance
    to :meth:`.CacheRegion.configure` to override default region invalidation.

    Example::

        class CustomInvalidationStrategy(RegionInvalidationStrategy):

            def __init__(self):
                self._soft_invalidated = None
                self._hard_invalidated = None

            def invalidate(self, hard=None):
                if hard:
                    self._soft_invalidated = None
                    self._hard_invalidated = time.time()
                else:
                    self._soft_invalidated = time.time()
                    self._hard_invalidated = None

            def is_invalidated(self, timestamp):
                return ((self._soft_invalidated and
                         timestamp < self._soft_invalidated) or
                        (self._hard_invalidated and
                         timestamp < self._hard_invalidated))

            def was_hard_invalidated(self):
                return bool(self._hard_invalidated)

            def is_hard_invalidated(self, timestamp):
                return (self._hard_invalidated and
                        timestamp < self._hard_invalidated)

            def was_soft_invalidated(self):
                return bool(self._soft_invalidated)

            def is_soft_invalidated(self, timestamp):
                return (self._soft_invalidated and
                        timestamp < self._soft_invalidated)

    The custom implementation is injected into a :class:`.CacheRegion`
    at configure time using the
    :paramref:`.CacheRegion.configure.region_invalidator` parameter::

        region = CacheRegion()

        region = region.configure(region_invalidator=CustomInvalidationStrategy())  # noqa

    Invalidation strategies that wish to have access to the
    :class:`.CacheRegion` itself should construct the invalidator given the
    region as an argument::

        class MyInvalidator(RegionInvalidationStrategy):
            def __init__(self, region):
                self.region = region
                # ...

            # ...

        region = CacheRegion()
        region = region.configure(region_invalidator=MyInvalidator(region))

    .. versionadded:: 0.6.2

    .. seealso::

        :paramref:`.CacheRegion.configure.region_invalidator`

    """

    def invalidate(self, hard: bool=True) -> None:
        """Region invalidation.

        :class:`.CacheRegion` propagated call.
        The default invalidation system works by setting
        a current timestamp (using ``time.time()``) to consider all older
        timestamps effectively invalidated.

        """
        raise NotImplementedError()

    def is_hard_invalidated(self, timestamp: float) -> bool:
        """Check timestamp to determine if it was hard invalidated.

        :return: Boolean. True if ``timestamp`` is older than
         the last region invalidation time and region is invalidated
         in hard mode.

        """
        raise NotImplementedError()

    def is_soft_invalidated(self, timestamp: float) -> bool:
        """Check timestamp to determine if it was soft invalidated.

        :return: Boolean. True if ``timestamp`` is older than
         the last region invalidation time and region is invalidated
         in soft mode.

        """
        raise NotImplementedError()

    def is_invalidated(self, timestamp: float) -> bool:
        """Check timestamp to determine if it was invalidated.

        :return: Boolean. True if ``timestamp`` is older than
         the last region invalidation time.

        """
        raise NotImplementedError()

    def was_soft_invalidated(self) -> bool:
        """Indicate the region was invalidated in soft mode.

        :return: Boolean. True if region was invalidated in soft mode.

        """
        raise NotImplementedError()

    def was_hard_invalidated(self) -> bool:
        """Indicate the region was invalidated in hard mode.

        :return: Boolean. True if region was invalidated in hard mode.

        """
        raise NotImplementedError()