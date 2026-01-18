import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
@util.positional(1)
def _get_multistore(filename, warn_on_readonly=True):
    """A helper method to initialize the multistore with proper locking.

    Args:
        filename: The JSON file storing a set of credentials
        warn_on_readonly: if True, log a warning if the store is readonly

    Returns:
        A multistore object
    """
    filename = os.path.expanduser(filename)
    _multistores_lock.acquire()
    try:
        multistore = _multistores.setdefault(filename, _MultiStore(filename, warn_on_readonly=warn_on_readonly))
    finally:
        _multistores_lock.release()
    return multistore