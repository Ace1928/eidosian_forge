from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import sys
import threading
import time
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.tools import analytics
Wait for up to `timeout` seconds for all error sources to finish.

    Preferentially raise "interesting" errors (errors not in the
    _UNINTERESTING_ERRORS) set.

    Args:
      timeout_sec: Seconds to wait for other error sources.
    