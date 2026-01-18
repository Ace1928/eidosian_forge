from the command line:
import functools
import re
import types
import unittest
import uuid
def _NonStringIterable(obj):
    return isinstance(obj, collections_abc.Iterable) and (not isinstance(obj, str))