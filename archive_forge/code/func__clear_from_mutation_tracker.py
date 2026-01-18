from __future__ import annotations
import gc
import atexit
import asyncio
import contextlib
import collections.abc
from lazyops.utils.lazy import lazy_import, get_keydb_enabled
from lazyops.utils.logs import logger, null_logger
from lazyops.utils.pooler import ThreadPooler
from typing import Any, Dict, Optional, Union, Iterable, List, Type, Set, Callable, Mapping, MutableMapping, Tuple, TypeVar, overload, TYPE_CHECKING
from .backends import LocalStatefulBackend, RedisStatefulBackend, StatefulBackendT
from .serializers import ObjectValue
from .addons import (
from .debug import get_autologger
def _clear_from_mutation_tracker(self, key: KT):
    """
        Clears the Mutation Tracker
        """
    _ = self._mutation_tracker.pop(key, None)
    _ = self._mutation_hashes.pop(key, None)