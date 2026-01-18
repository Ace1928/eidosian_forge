import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
class StudentTWithAbsDfSoftplusScale(StudentT):
    """StudentT with `df = floor(abs(df))` and `scale = softplus(scale)`."""

    @deprecation.deprecated('2019-01-01', 'Use `tfd.StudentT(tf.floor(tf.abs(df)), loc, tf.nn.softplus(scale)) instead.', warn_once=True)
    def __init__(self, df, loc, scale, validate_args=False, allow_nan_stats=True, name='StudentTWithAbsDfSoftplusScale'):
        parameters = dict(locals())
        with ops.name_scope(name, values=[df, scale]) as name:
            super(StudentTWithAbsDfSoftplusScale, self).__init__(df=math_ops.floor(math_ops.abs(df)), loc=loc, scale=nn.softplus(scale, name='softplus_scale'), validate_args=validate_args, allow_nan_stats=allow_nan_stats, name=name)
        self._parameters = parameters