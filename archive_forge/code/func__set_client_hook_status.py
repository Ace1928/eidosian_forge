import os
import threading
from contextlib import contextmanager
from functools import wraps
from ray._private.auto_init_hook import auto_init_ray
def _set_client_hook_status(val: bool):
    global _client_hook_status_on_thread
    _client_hook_status_on_thread.status = val