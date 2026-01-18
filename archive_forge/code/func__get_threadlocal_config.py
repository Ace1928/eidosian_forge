import os
import threading
from contextlib import contextmanager as contextmanager
def _get_threadlocal_config():
    """Get a threadlocal **mutable** configuration. If the configuration
    does not exist, copy the default global configuration."""
    if not hasattr(_threadlocal, 'global_config'):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config