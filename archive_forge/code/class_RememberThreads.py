import threading
import unittest
from unittest import mock
from traits.api import Float, HasTraits, List
from traits.testing.unittest_tools import UnittestTools
class RememberThreads(object):
    """
    Context manager that behaves like Thread, but remembers created
    threads so that they can be joined.
    """

    def __init__(self):
        self._threads = []

    def __call__(self, *args, **kwargs):
        thread = threading.Thread(*args, **kwargs)
        self._threads.append(thread)
        return thread

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        threads = self._threads
        while threads:
            thread = threads.pop()
            thread.join(timeout=SAFETY_TIMEOUT)
            if thread.is_alive():
                raise RuntimeError('Failed to join thread')