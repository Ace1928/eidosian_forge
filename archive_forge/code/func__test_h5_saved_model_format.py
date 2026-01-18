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
def _test_h5_saved_model_format(f, test_or_class, *args, **kwargs):
    with testing_utils.saved_model_format_scope('h5'):
        f(test_or_class, *args, **kwargs)