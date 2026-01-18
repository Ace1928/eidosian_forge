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
def _reraise_if_subprocess_error(self, process_statuses):
    for process_status in process_statuses.values():
        assert isinstance(process_status, _ProcessStatusInfo)
        if not process_status.is_successful:
            process_status.exc_info[1].mpr_result = self._get_mpr_result(process_statuses)
            six.reraise(*process_status.exc_info)