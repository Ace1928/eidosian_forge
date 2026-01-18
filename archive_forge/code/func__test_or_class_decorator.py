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
def _test_or_class_decorator(test_or_class, single_method_decorator):
    """Decorate a test or class with a decorator intended for one method.

  If the test_or_class is a class:
    This will apply the decorator to all test methods in the class.

  If the test_or_class is an iterable of already-parameterized test cases:
    This will apply the decorator to all the cases, and then flatten the
    resulting cross-product of test cases. This allows stacking the Keras
    parameterized decorators w/ each other, and to apply them to test methods
    that have already been marked with an absl parameterized decorator.

  Otherwise, treat the obj as a single method and apply the decorator directly.

  Args:
    test_or_class: A test method (that may have already been decorated with a
      parameterized decorator, or a test class that extends
      keras_parameterized.TestCase
    single_method_decorator:
      A parameterized decorator intended for a single test method.
  Returns:
    The decorated result.
  """

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
    if test_or_class is not None:
        return _decorate_test_or_class(test_or_class)
    return _decorate_test_or_class