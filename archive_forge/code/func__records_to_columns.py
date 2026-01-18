import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import na_values
from rpy2.rinterface import IntSexpVector
from rpy2.rinterface import ListSexpVector
from rpy2.rinterface import SexpVector
from rpy2.rinterface import StrSexpVector
import datetime
import functools
import math
import numpy  # type: ignore
import pandas  # type: ignore
import pandas.core.series  # type: ignore
from pandas.core.frame import DataFrame as PandasDataFrame  # type: ignore
from pandas.core.dtypes.api import is_datetime64_any_dtype  # type: ignore
import warnings
from collections import OrderedDict
from rpy2.robjects.vectors import (BoolVector,
import rpy2.robjects.numpy2ri as numpy2ri
def _records_to_columns(obj):
    columns = OrderedDict()
    obj_ri = ListSexpVector(obj)
    for i, record in enumerate(obj_ri):
        checknames = set()
        for name in record.names:
            if name in checknames:
                raise ValueError(f'The record {i} has "{name}" duplicated.')
            checknames.add(name)
            if name not in columns:
                columns[name] = []
    columnnames = set(columns.keys())
    for i, record in enumerate(obj_ri):
        checknames = set()
        for name, value in zip(record.names, record):
            checknames.add(name)
            if hasattr(value, '__len__'):
                if len(value) != 1:
                    raise ValueError(f'The value for "{name}" record {i} is not a scalar. It has {len(value)} elements.')
                else:
                    value = value[0]
            columns[name].append(value)
        for name in columnnames - checknames:
            columns[name].append(None)
    return columns