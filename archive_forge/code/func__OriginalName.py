from the command line:
import functools
import re
import types
import unittest
import uuid
def _OriginalName(self):
    return self._testMethodName.split(_SEPARATOR)[0]