import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
def _get_min_distance_numpy(element):
    """
    NumPy based implementation of get_min_distance
    """
    xys = element.array([0, 1])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in')
        xys = xys.astype('float32').view(np.complex64)
        distances = np.abs(xys.T - xys)
        np.fill_diagonal(distances, np.inf)
        distances = distances[distances > 0]
        if len(distances):
            return distances.min()
    return 0