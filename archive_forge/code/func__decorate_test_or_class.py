import collections
import functools
import itertools
import unittest
from absl.testing import parameterized
from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.util import nest
def _decorate_test_or_class(obj):
    if isinstance(obj, collections.abc.Iterable):
        return itertools.chain.from_iterable((single_method_decorator(method) for method in obj))
    if isinstance(obj, type):
        cls = obj
        for name, value in cls.__dict__.copy().items():
            if callable(value) and name.startswith(unittest.TestLoader.testMethodPrefix):
                setattr(cls, name, single_method_decorator(value))
        cls = type(cls).__new__(type(cls), cls.__name__, cls.__bases__, cls.__dict__.copy())
        return cls
    return single_method_decorator(obj)