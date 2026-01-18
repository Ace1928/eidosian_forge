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
def _add_any_metadata(table, pandas_metadata):
    modified_columns = {}
    modified_fields = {}
    schema = table.schema
    index_columns = pandas_metadata['index_columns']
    index_columns = [idx_col for idx_col in index_columns if isinstance(idx_col, str)]
    n_index_levels = len(index_columns)
    n_columns = len(pandas_metadata['columns']) - n_index_levels
    for i, col_meta in enumerate(pandas_metadata['columns']):
        raw_name = col_meta.get('field_name')
        if not raw_name:
            raw_name = col_meta['name']
            if i >= n_columns:
                raw_name = index_columns[i - n_columns]
            if raw_name is None:
                raw_name = 'None'
        idx = schema.get_field_index(raw_name)
        if idx != -1:
            if col_meta['pandas_type'] == 'datetimetz':
                col = table[idx]
                if not isinstance(col.type, pa.lib.TimestampType):
                    continue
                metadata = col_meta['metadata']
                if not metadata:
                    continue
                metadata_tz = metadata.get('timezone')
                if metadata_tz and metadata_tz != col.type.tz:
                    converted = col.to_pandas()
                    tz_aware_type = pa.timestamp('ns', tz=metadata_tz)
                    with_metadata = pa.Array.from_pandas(converted, type=tz_aware_type)
                    modified_fields[idx] = pa.field(schema[idx].name, tz_aware_type)
                    modified_columns[idx] = with_metadata
    if len(modified_columns) > 0:
        columns = []
        fields = []
        for i in range(len(table.schema)):
            if i in modified_columns:
                columns.append(modified_columns[i])
                fields.append(modified_fields[i])
            else:
                columns.append(table[i])
                fields.append(table.schema[i])
        return pa.Table.from_arrays(columns, schema=pa.schema(fields))
    else:
        return table