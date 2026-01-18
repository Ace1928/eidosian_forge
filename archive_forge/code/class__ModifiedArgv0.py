importers when locating support scripts as well as when importing modules.
import sys
import importlib.machinery # importlib first so we can test #15386 via -m
import importlib.util
import io
import os
class _ModifiedArgv0(object):

    def __init__(self, value):
        self.value = value
        self._saved_value = self._sentinel = object()

    def __enter__(self):
        if self._saved_value is not self._sentinel:
            raise RuntimeError('Already preserving saved value')
        self._saved_value = sys.argv[0]
        sys.argv[0] = self.value

    def __exit__(self, *args):
        self.value = self._sentinel
        sys.argv[0] = self._saved_value