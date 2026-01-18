import logging
from gzip import GzipFile
from os import PathLike, makedirs, remove
from os.path import exists, join
import joblib
import numpy as np
import scipy.sparse as sp
from ..utils import Bunch
from ..utils import shuffle as shuffle_
from ..utils._param_validation import StrOptions, validate_params
from . import get_data_home
from ._base import RemoteFileMetadata, _fetch_remote, _pkl_filepath, load_descr
from ._svmlight_format_io import load_svmlight_files
def _find_permutation(a, b):
    """Find the permutation from a to b."""
    t = np.argsort(a)
    u = np.argsort(b)
    u_ = _inverse_permutation(u)
    return t[u_]