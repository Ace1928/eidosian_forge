import io
import numpy as np
import pyarrow as pa
from pyarrow.tests import util
def _range_integers(size, dtype):
    return pa.array(np.arange(size, dtype=dtype))