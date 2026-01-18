import itertools
from contextlib import ExitStack
class RecordingContextManager:
    """A context manager that records."""

    def __init__(self):
        self._calls = []

    def __enter__(self):
        self._calls.append('__enter__')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._calls.append('__exit__')
        return False