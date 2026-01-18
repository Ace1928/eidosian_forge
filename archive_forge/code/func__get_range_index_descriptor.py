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
def _get_range_index_descriptor(level):
    return {'kind': 'range', 'name': _level_name(level.name), 'start': _pandas_api.get_rangeindex_attribute(level, 'start'), 'stop': _pandas_api.get_rangeindex_attribute(level, 'stop'), 'step': _pandas_api.get_rangeindex_attribute(level, 'step')}