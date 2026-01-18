from collections import OrderedDict
import contextlib
import re
import types
import unittest
from absl.testing import parameterized
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.test.combinations.NamedObject', v1=[])
class NamedObject:
    """A class that translates an object into a good test name."""

    def __init__(self, name, obj):
        self._name = name
        self._obj = obj

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __iter__(self):
        return self._obj.__iter__()

    def __repr__(self):
        return self._name