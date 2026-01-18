import numpy
from rasterio.env import GDALVersion
def _getnpdtype(dtype):
    import numpy as np
    if _is_complex_int(dtype):
        return np.dtype('complex64')
    else:
        return np.dtype(dtype)