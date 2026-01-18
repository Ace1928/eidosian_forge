from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
def _randomUniformAvoidAnchors(self, low, high, anchors, radius, num_samples):
    """Generate samples that are far enough from a set of anchor points.

    We generate uniform samples in [low, high], then reject those that are less
    than radius away from any point in anchors. We stop after we have accepted
    num_samples samples.

    Args:
      low: The lower end of the interval.
      high: The upper end of the interval.
      anchors: A list of length num_crops with anchor points to avoid.
      radius: Distance threshold for the samples from the anchors.
      num_samples: How many samples to produce.

    Returns:
      samples: A list of length num_samples with the accepted samples.
    """
    self.assertTrue(low < high)
    self.assertTrue(radius >= 0)
    num_anchors = len(anchors)
    self.assertTrue(2 * radius * num_anchors < 0.5 * (high - low))
    anchors = np.reshape(anchors, num_anchors)
    samples = []
    while len(samples) < num_samples:
        sample = np.random.uniform(low, high)
        if np.all(np.fabs(sample - anchors) > radius):
            samples.append(sample)
    return samples