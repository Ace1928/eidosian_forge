import collections
import glob
import json
import os
import platform
import re
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def _calculate_t0(self):
    """Calculate the first timestamp across all devices."""
    t0s = [t0 for t0 in self._t0s.values() if t0 is not None]
    self._t0 = min(t0s) if t0s else None