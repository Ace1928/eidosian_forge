from the command line:
import functools
import re
import types
import unittest
import uuid
def _StrClass(cls):
    return '%s.%s' % (cls.__module__, cls.__name__)