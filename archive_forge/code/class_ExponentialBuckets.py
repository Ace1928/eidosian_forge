import collections
import functools
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
class ExponentialBuckets(Buckets):
    """Exponential bucketing strategy.

  Sets up buckets of the form:
      [-DBL_MAX, ..., scale * growth^i,
       scale * growth_factor^(i + 1), ..., DBL_MAX].
  """
    __slots__ = []

    def __init__(self, scale, growth_factor, bucket_count):
        """Creates a new exponential Buckets.

    Args:
      scale: float
      growth_factor: float
      bucket_count: integer
    """
        super(ExponentialBuckets, self).__init__(pywrap_tfe.TFE_MonitoringNewExponentialBuckets(scale, growth_factor, bucket_count))