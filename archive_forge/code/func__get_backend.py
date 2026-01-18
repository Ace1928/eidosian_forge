import base64
import json
import logging
import os
import threading
import fasteners
from six import iteritems
from oauth2client import _helpers
from oauth2client import client
def _get_backend(filename):
    """A helper method to get or create a backend with thread locking.

    This ensures that only one backend is used per-file per-process, so that
    thread and process locks are appropriately shared.

    Args:
        filename: The full path to the credential storage file.

    Returns:
        An instance of :class:`_MultiprocessStorageBackend`.
    """
    filename = os.path.abspath(filename)
    with _backends_lock:
        if filename not in _backends:
            _backends[filename] = _MultiprocessStorageBackend(filename)
        return _backends[filename]