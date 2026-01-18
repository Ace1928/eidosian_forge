from __future__ import print_function
import os
import socket
import signal
import threading
from contextlib import closing, contextmanager
from . import _gi
Installs a SIGINT signal handler in case the default Python one is
    active which calls 'callback' in case the signal occurs.

    Only does something if called from the main thread.

    In case of nested context managers the signal handler will be only
    installed once and the callbacks will be called in the reverse order
    of their registration.

    The old signal handler will be restored in case no signal handler is
    registered while the context is active.
    