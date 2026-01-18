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
class FragmentStorage(Protocol):
    """A key-value store for Fragments. Used to implement the @st.experimental_fragment
    decorator.

    We intentionally define this as its own protocol despite how generic it appears to
    be at first glance. The reason why is that, in any case where fragments aren't just
    stored as Python closures in memory, storing and retrieving Fragments will generally
    involve serializing and deserializing function bytecode, which is a tricky aspect
    to implementing FragmentStorages that won't generally appear with our other *Storage
    protocols.
    """

    @abstractmethod
    def get(self, key: str) -> Fragment:
        """Returns the stored fragment for the given key."""
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Fragment) -> None:
        """Saves a fragment under the given key."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the fragment corresponding to the given key."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all fragments saved in this FragmentStorage."""
        raise NotImplementedError