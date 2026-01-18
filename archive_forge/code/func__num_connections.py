import io
import os
import socket
import threading
import time
import selectors
from contextlib import suppress
from . import errors
from ._compat import IS_WINDOWS
from .makefile import MakeFile
@property
def _num_connections(self):
    """Return the current number of connections.

        Includes all connections registered with the selector,
        minus one for the server socket, which is always registered
        with the selector.
        """
    return len(self._selector) - 1