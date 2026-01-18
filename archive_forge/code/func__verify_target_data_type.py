import gzip
import hashlib
import json
import os
import shutil
import time
from contextlib import closing
from functools import wraps
from os.path import join
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from warnings import warn
import numpy as np
from ..utils import (
from ..utils._param_validation import (
from . import get_data_home
from ._arff_parser import load_arff_from_gzip_file
def _verify_target_data_type(features_dict, target_columns):
    if not isinstance(target_columns, list):
        raise ValueError('target_column should be list, got: %s' % type(target_columns))
    found_types = set()
    for target_column in target_columns:
        if target_column not in features_dict:
            raise KeyError(f"Could not find target_column='{target_column}'")
        if features_dict[target_column]['data_type'] == 'numeric':
            found_types.add(np.float64)
        else:
            found_types.add(object)
        if features_dict[target_column]['is_ignore'] == 'true':
            warn(f"target_column='{target_column}' has flag is_ignore.")
        if features_dict[target_column]['is_row_identifier'] == 'true':
            warn(f"target_column='{target_column}' has flag is_row_identifier.")
    if len(found_types) > 1:
        raise ValueError('Can only handle homogeneous multi-target datasets, i.e., all targets are either numeric or categorical.')