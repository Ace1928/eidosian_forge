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
def _close_streaming(self):
    """Close stdout, stderr and streaming pipe.

    We need to explicitly close them since Tensorflow may take a while to exit,
    so that the reading threads in the main process can exit more quickly.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout.close()
    sys.stderr.close()
    self._resources.streaming_pipe_w.close()