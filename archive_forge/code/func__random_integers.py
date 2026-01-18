import io
import numpy as np
import pyarrow as pa
from pyarrow.tests import util
def _random_integers(size, dtype):
    platform_int_info = np.iinfo('int_')
    iinfo = np.iinfo(dtype)
    return np.random.randint(max(iinfo.min, platform_int_info.min), min(iinfo.max, platform_int_info.max), size=size).astype(dtype)