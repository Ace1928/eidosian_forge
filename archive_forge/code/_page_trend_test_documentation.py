from itertools import permutations
import numpy as np
import math
from ._continuous_distns import norm
import scipy.stats
from dataclasses import dataclass
Recursive function to evaluate p(l, k, n); see [5] Equation 1