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
def _check_data_column_metadata_consistency(all_columns):
    assert all((c['name'] is None and 'field_name' in c or c['name'] is not None for c in all_columns))