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
def _replace_value(self, data):
    """Replace `nodata` value in data with NaN, if specified in opts"""
    data = data.astype('float64')
    mask = data != self.p.nodata
    if hasattr(data, 'where'):
        return data.where(mask, np.nan)
    return np.where(mask, data, np.nan)