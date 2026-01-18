import logging
from io import BytesIO
from os import PathLike, makedirs, remove
from os.path import exists
import joblib
import numpy as np
from ..utils import Bunch
from ..utils._param_validation import validate_params
from . import get_data_home
from ._base import RemoteFileMetadata, _fetch_remote, _pkl_filepath
def _load_coverage(F, header_length=6, dtype=np.int16):
    """Load a coverage file from an open file object.

    This will return a numpy array of the given dtype
    """
    header = [F.readline() for _ in range(header_length)]
    make_tuple = lambda t: (t.split()[0], float(t.split()[1]))
    header = dict([make_tuple(line) for line in header])
    M = np.loadtxt(F, dtype=dtype)
    nodata = int(header[b'NODATA_value'])
    if nodata != -9999:
        M[nodata] = -9999
    return M