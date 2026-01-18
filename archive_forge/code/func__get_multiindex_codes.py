import ast
from collections.abc import Sequence
from concurrent import futures
import concurrent.futures.thread  # noqa
from copy import deepcopy
from itertools import zip_longest
import json
import operator
import re
import warnings
import numpy as np
import pyarrow as pa
from pyarrow.lib import _pandas_api, frombytes  # noqa
def _get_multiindex_codes(mi):
    if isinstance(mi, _pandas_api.pd.MultiIndex):
        return mi.codes
    else:
        return None