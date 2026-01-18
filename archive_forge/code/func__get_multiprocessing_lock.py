from __future__ import annotations
import multiprocessing
import threading
import uuid
import weakref
from collections.abc import Hashable, MutableMapping
from typing import Any, ClassVar
from weakref import WeakValueDictionary
def _get_multiprocessing_lock(key):
    del key
    return multiprocessing.Lock()