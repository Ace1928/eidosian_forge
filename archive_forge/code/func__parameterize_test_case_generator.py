import itertools
import types
import unittest
from cupy.testing import _bundle
from cupy.testing import _pytest_impl
def _parameterize_test_case_generator(base, params):
    for i, param in enumerate(params):
        yield _parameterize_test_case(base, i, param)