import time
from tensorflow.python.eager import monitoring
from tensorflow.python.util import tf_contextlib
def _get_metric_histogram(histogram_proto):
    """Convert a histogram proto into a dict.

  Args:
    histogram_proto: a proto containing a Sampler metric's result histogram.

  Returns:
    A dict containing summary statistics and the raw histogram values.
  """
    ret = dict()
    ret['min'] = histogram_proto.min
    ret['max'] = histogram_proto.max
    ret['num'] = histogram_proto.num
    ret['sum'] = histogram_proto.sum
    bucket_limits = histogram_proto.bucket_limit
    bucket_vals = histogram_proto.bucket
    ret['histogram'] = {}
    bucket_limits.insert(0, 0)
    for lb, ub, val in zip(bucket_limits[:-1], bucket_limits[1:], bucket_vals):
        ret['histogram'][lb, ub] = val
    return ret