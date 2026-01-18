import errno
import logging
import os
import threading
import time
import six
from fasteners import _utils
def _ensure_tree(path):
    """Create a directory (and any ancestor directories required).

    :param path: Directory to create
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            if not os.path.isdir(path):
                raise
            else:
                return False
        elif e.errno == errno.EISDIR:
            return False
        else:
            raise
    else:
        return True