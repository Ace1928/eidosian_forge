import csv
import gzip
import hashlib
import os
import shutil
from collections import namedtuple
from importlib import resources
from numbers import Integral
from os import environ, listdir, makedirs
from os.path import expanduser, isdir, join, splitext
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
from ..preprocessing import scale
from ..utils import Bunch, check_pandas_support, check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
def _convert_data_dataframe(caller_name, data, target, feature_names, target_names, sparse_data=False):
    pd = check_pandas_support('{} with as_frame=True'.format(caller_name))
    if not sparse_data:
        data_df = pd.DataFrame(data, columns=feature_names, copy=False)
    else:
        data_df = pd.DataFrame.sparse.from_spmatrix(data, columns=feature_names)
    target_df = pd.DataFrame(target, columns=target_names)
    combined_df = pd.concat([data_df, target_df], axis=1)
    X = combined_df[feature_names]
    y = combined_df[target_names]
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    return (combined_df, X, y)