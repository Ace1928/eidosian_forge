import math
import numpy as np
from tensorflow.python.ops.distributions import special_math
def _normal_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))