from __future__ import unicode_literals
import inspect
import os
import signal
import sys
import threading
import weakref
from wcwidth import wcwidth
from six.moves import range
class DummyContext(object):
    """
    (contextlib.nested is not available on Py3)
    """

    def __enter__(self):
        pass

    def __exit__(self, *a):
        pass