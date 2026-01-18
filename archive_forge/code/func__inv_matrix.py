import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
@property
def _inv_matrix(self):
    return np.linalg.inv(self.params)