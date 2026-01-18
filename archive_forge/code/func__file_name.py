import gzip
import json
import os
import re
from functools import partial
from importlib import resources
from io import BytesIO
from urllib.error import HTTPError
import numpy as np
import pytest
import scipy.sparse
import sklearn
from sklearn import config_context
from sklearn.datasets import fetch_openml as fetch_openml_orig
from sklearn.datasets._openml import (
from sklearn.utils import Bunch, check_pandas_support
from sklearn.utils._testing import (
def _file_name(url, suffix):
    output = re.sub('\\W', '-', url[len('https://api.openml.org/'):]) + suffix + path_suffix
    return output.replace('-json-data-list', '-jdl').replace('-json-data-features', '-jdf').replace('-json-data-qualities', '-jdq').replace('-json-data', '-jd').replace('-data_name', '-dn').replace('-download', '-dl').replace('-limit', '-l').replace('-data_version', '-dv').replace('-status', '-s').replace('-deactivated', '-dact').replace('-active', '-act')