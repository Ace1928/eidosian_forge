from __future__ import annotations
import contextlib
import hashlib
import inspect
from abc import abstractmethod
from copy import deepcopy
from datetime import timedelta
from functools import wraps
from typing import Any, Callable, Protocol, TypeVar, overload
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.time_util import time_to_seconds
class MemoryFragmentStorage(FragmentStorage):
    """A simple, memory-backed implementation of FragmentStorage.

    MemoryFragmentStorage is just a wrapper around a plain Python dict that complies with
    the FragmentStorage protocol.
    """

    def __init__(self):
        self._fragments: dict[str, Fragment] = {}

    def get(self, key: str) -> Fragment:
        return self._fragments[key]

    def set(self, key: str, value: Fragment) -> None:
        self._fragments[key] = value

    def delete(self, key: str) -> None:
        del self._fragments[key]

    def clear(self) -> None:
        self._fragments.clear()