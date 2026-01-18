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
class DefaultInvalidationStrategy(RegionInvalidationStrategy):

    def __init__(self):
        self._is_hard_invalidated = None
        self._invalidated = None

    def invalidate(self, hard: bool=True) -> None:
        self._is_hard_invalidated = bool(hard)
        self._invalidated = time.time()

    def is_invalidated(self, timestamp: float) -> bool:
        return self._invalidated is not None and timestamp < self._invalidated

    def was_hard_invalidated(self) -> bool:
        return self._is_hard_invalidated is True

    def is_hard_invalidated(self, timestamp: float) -> bool:
        return self.was_hard_invalidated() and self.is_invalidated(timestamp)

    def was_soft_invalidated(self) -> bool:
        return self._is_hard_invalidated is False

    def is_soft_invalidated(self, timestamp: float) -> bool:
        return self.was_soft_invalidated() and self.is_invalidated(timestamp)