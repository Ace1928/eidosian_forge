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
def _resolve_columns_of_interest(df, schema, columns):
    if schema is not None and columns is not None:
        raise ValueError('Schema and columns arguments are mutually exclusive, pass only one of them')
    elif schema is not None:
        columns = schema.names
    elif columns is not None:
        columns = [c for c in columns if c in df.columns]
    else:
        columns = df.columns
    return columns