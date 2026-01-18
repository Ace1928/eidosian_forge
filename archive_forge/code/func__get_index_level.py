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
def _get_index_level(df, name):
    """
    Get the index level of a DataFrame given 'name' (column name in an arrow
    Schema).
    """
    key = name
    if name not in df.index.names and _is_generated_index_name(name):
        key = int(name[len('__index_level_'):-2])
    return df.index.get_level_values(key)