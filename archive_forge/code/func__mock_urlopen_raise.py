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
def _mock_urlopen_raise(request, *args, **kwargs):
    raise ValueError('This mechanism intends to test correct cachehandling. As such, urlopen should never be accessed. URL: %s' % request.get_full_url())