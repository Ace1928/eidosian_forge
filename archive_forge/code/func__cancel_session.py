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
def _cancel_session():
    time.sleep(5)
    tf.compat.v1.logging.error('Closing session due to error %s' % value)
    try:
        session.close()
    except:
        tf.compat.v1.logging.error('\n\n\nFailed to close session after error.Other threads may hang.\n\n\n')