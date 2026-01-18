from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import os
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _validate_delay(delay):
    """Check that delay is an integer value.

    Since this has to work for both Python2 and Python3 and PEP237 defines long
    to be basically int, we cannot just use a lambda function.
    """
    try:
        return isinstance(delay, (int, long))
    except NameError:
        return isinstance(delay, int)