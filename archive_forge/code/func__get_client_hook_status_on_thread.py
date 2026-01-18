import os
import threading
from contextlib import contextmanager
from functools import wraps
from ray._private.auto_init_hook import auto_init_ray
def _get_client_hook_status_on_thread():
    """Get's the value of `_client_hook_status_on_thread`.
    Since `_client_hook_status_on_thread` is a thread-local variable, we may
    need to add and set the 'status' attribute.
    """
    global _client_hook_status_on_thread
    if not hasattr(_client_hook_status_on_thread, 'status'):
        _client_hook_status_on_thread.status = True
    return _client_hook_status_on_thread.status