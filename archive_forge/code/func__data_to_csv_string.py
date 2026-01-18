import json
import os
import random
import hashlib
import warnings
from typing import Union, MutableMapping, Optional, Dict, Sequence, TYPE_CHECKING, List
import pandas as pd
from toolz import curried
from typing import TypeVar
from ._importers import import_pyarrow_interchange
from .core import sanitize_dataframe, sanitize_arrow_table, DataFrameLike
from .core import sanitize_geo_interface
from .deprecation import AltairDeprecationWarning
from .plugin_registry import PluginRegistry
from typing import Protocol, TypedDict, Literal
def _data_to_csv_string(data: Union[dict, pd.DataFrame, DataFrameLike]) -> str:
    """return a CSV string representation of the input data"""
    check_data_type(data)
    if hasattr(data, '__geo_interface__'):
        raise NotImplementedError('to_csv does not work with data that contains the __geo_interface__ attribute')
    elif isinstance(data, pd.DataFrame):
        data = sanitize_dataframe(data)
        return data.to_csv(index=False)
    elif isinstance(data, dict):
        if 'values' not in data:
            raise KeyError('values expected in data dict, but not present')
        return pd.DataFrame.from_dict(data['values']).to_csv(index=False)
    elif isinstance(data, DataFrameLike):
        import pyarrow as pa
        import pyarrow.csv as pa_csv
        pa_table = arrow_table_from_dfi_dataframe(data)
        csv_buffer = pa.BufferOutputStream()
        pa_csv.write_csv(pa_table, csv_buffer)
        return csv_buffer.getvalue().to_pybytes().decode()
    else:
        raise NotImplementedError('to_csv only works with data expressed as a DataFrame or as a dict')