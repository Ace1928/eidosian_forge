import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
Read the root pages.

        This is structured as a generator, so that the root records can be
        yielded up to whoever needs them without any buffering.
        