import os
import threading
from contextlib import contextmanager
from functools import wraps
from ray._private.auto_init_hook import auto_init_ray
def _disable_client_hook():
    global _client_hook_status_on_thread
    out = _get_client_hook_status_on_thread()
    _client_hook_status_on_thread.status = False
    return out