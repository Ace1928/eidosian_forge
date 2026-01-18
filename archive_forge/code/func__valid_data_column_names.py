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
def _valid_data_column_names(features_list, target_columns):
    valid_data_column_names = []
    for feature in features_list:
        if feature['name'] not in target_columns and feature['is_ignore'] != 'true' and (feature['is_row_identifier'] != 'true'):
            valid_data_column_names.append(feature['name'])
    return valid_data_column_names