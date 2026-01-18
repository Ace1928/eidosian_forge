import logging
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from .. import utils
from ..rcparams import rcParams
from .base import CoordSpec, DimSpec, dict_to_dataset, infer_stan_dtypes, requires
from .inference_data import InferenceData
def _unpack_ndarrays(arrays, columns, dtypes=None):
    """Transform a list of ndarrays to dictionary containing ndarrays.

    Parameters
    ----------
    arrays : List[np.ndarray]
    columns: Dict[str, int]
    dtypes: Dict[str, Any]

    Returns
    -------
    Dict
        key, values pairs. Values are formatted to shape = (nchain, ndraws, *shape)
    """
    col_groups = defaultdict(list)
    for col, col_idx in columns.items():
        key, *loc = col.split('.')
        loc = tuple((int(i) - 1 for i in loc))
        col_groups[key].append((col_idx, loc))
    chains = len(arrays)
    draws = len(arrays[0])
    sample = {}
    if draws:
        for key, cols_locs in col_groups.items():
            ndim = np.array([loc for _, loc in cols_locs]).max(0) + 1
            dtype = dtypes.get(key, np.float64)
            sample[key] = np.zeros((chains, draws, *ndim), dtype=dtype)
            for col, loc in cols_locs:
                for chain_id, arr in enumerate(arrays):
                    draw = arr[:, col]
                    if loc == ():
                        sample[key][chain_id, :] = draw
                    else:
                        axis1_all = range(sample[key].shape[1])
                        slicer = (chain_id, axis1_all, *loc)
                        sample[key][slicer] = draw
    return sample