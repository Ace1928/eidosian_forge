import collections
import functools
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
class Buckets(object):
    """Bucketing strategies for the samplers."""
    __slots__ = ['buckets']

    def __init__(self, buckets):
        """Creates a new Buckets.

    Args:
      buckets: A c pointer of TFE_MonitoringBuckets.
    """
        self.buckets = buckets

    def __del__(self):
        pywrap_tfe.TFE_MonitoringDeleteBuckets(self.buckets)