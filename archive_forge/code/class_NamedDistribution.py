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
class NamedDistribution(object):
    """Wraps a `tf.distribute.Strategy` and adds a name for test titles."""

    def __init__(self, name, distribution_fn, required_gpus=None, required_physical_gpus=0, required_tpu=False, use_cloud_tpu=False, has_chief=False, num_workers=1, num_ps=0, share_gpu=True, pool_runner_fn=None, no_xla=False):
        """Initialize NamedDistribution.

    Args:
      name: Name that will be a part of the name of the test case.
      distribution_fn: A callable that creates a `tf.distribute.Strategy`.
      required_gpus: The number of GPUs that the strategy requires. Only one of
      `required_gpus` and `required_physical_gpus` should be set.
      required_physical_gpus: Number of physical GPUs required. Only one of
      `required_gpus` and `required_physical_gpus` should be set.
      required_tpu: Whether the strategy requires TPU.
      use_cloud_tpu: Whether the strategy requires cloud TPU.
      has_chief: Whether the strategy requires a chief worker.
      num_workers: The number of workers that the strategy requires.
      num_ps: The number of parameter servers.
      share_gpu: Whether to share GPUs among workers.
      pool_runner_fn: An optional callable that returns a MultiProcessPoolRunner
        to run the test.
      no_xla: Whether to skip in XLA tests.
    """
        object.__init__(self)
        self._name = name
        self._distribution_fn = distribution_fn
        self.required_gpus = required_gpus
        self.required_physical_gpus = required_physical_gpus
        self.required_tpu = required_tpu
        self.use_cloud_tpu = use_cloud_tpu
        self.has_chief = has_chief
        self.num_workers = num_workers
        self.num_ps = num_ps
        self.share_gpu = share_gpu
        self._pool_runner_fn = pool_runner_fn
        self.no_xla = no_xla

    @property
    def runner(self):
        if self._pool_runner_fn is not None:
            return self._pool_runner_fn()
        return None

    @property
    def strategy(self):
        return self._distribution_fn()

    def __repr__(self):
        return self._name