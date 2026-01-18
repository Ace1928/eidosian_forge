import collections
import copy
import re
import sys
import types
import unittest
from absl import app
import six
from tensorflow.python.client import session
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations as framework_combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations as combinations_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
class TestEnvironment(object):
    """Holds the test environment information.

  Tests should modify the attributes of the instance returned by `env()` in the
  main process if needed, and it will be passed to the worker processes each
  time a test case is run.
  """

    def __init__(self):
        self.tf_data_service_dispatcher = None
        self.total_phsyical_gpus = None

    def __setattr__(self, name, value):
        if not in_main_process():
            raise ValueError('combinations.env() should only be modified in the main process. Condition your code on combinations.in_main_process().')
        super().__setattr__(name, value)