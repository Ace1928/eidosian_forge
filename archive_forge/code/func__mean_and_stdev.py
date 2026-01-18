import math
import numbers
import os
import re
import sys
import time
import types
from absl import app
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _mean_and_stdev(x):
    if not x:
        return (-1, -1)
    l = len(x)
    mean = sum(x) / l
    if l == 1:
        return (mean, -1)
    variance = sum([(e - mean) * (e - mean) for e in x]) / (l - 1)
    return (mean, math.sqrt(variance))