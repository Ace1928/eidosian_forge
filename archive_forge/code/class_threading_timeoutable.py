import ctypes
import threading
from .utils import TimeoutException, BaseTimeout, base_timeoutable
class threading_timeoutable(base_timeoutable):
    """A function or method decorator that raises a ``TimeoutException`` to
    decorated functions that should not last a certain amount of time.
    this one uses ``ThreadingTimeout`` context manager.

    See :class:`.utils.base_timoutable`` class for further comments.
    """
    to_ctx_mgr = ThreadingTimeout