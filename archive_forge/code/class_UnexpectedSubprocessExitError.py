import collections
import contextlib
import json
import os
import signal
import sys
import threading
import time
import unittest
import weakref
from absl import logging
import six
from six.moves import queue as Queue
from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.distribute.multi_process_runner.UnexpectedSubprocessExitError', v1=[])
class UnexpectedSubprocessExitError(RuntimeError):
    """An error indicating there is at least one subprocess with unexpected exit.

  When this is raised, a namedtuple object representing the multi-process run
  result can be retrieved by
  `tf.__internal__.distribute.multi_process_runner
  .UnexpectedSubprocessExitError`'s
  `mpr_result` attribute. See
  `tf.__internal__.distribute.multi_process_runner.run` for more information.
  """

    def __init__(self, msg, mpr_result):
        super(UnexpectedSubprocessExitError, self).__init__(msg)
        self.mpr_result = mpr_result