import collections
import csv
import functools
import gzip
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.experimental.ops import parsing_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import map_op
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util.tf_export import tf_export
def _infer_column_defaults(filenames, num_cols, field_delim, use_quote_delim, na_value, header, num_rows_for_inference, select_columns, file_io_fn):
    """Infers column types from the first N valid CSV records of files."""
    if select_columns is None:
        select_columns = range(num_cols)
    inferred_types = [None] * len(select_columns)
    for i, csv_row in enumerate(_next_csv_row(filenames, num_cols, field_delim, use_quote_delim, header, file_io_fn)):
        if num_rows_for_inference is not None and i >= num_rows_for_inference:
            break
        for j, col_index in enumerate(select_columns):
            inferred_types[j] = _infer_type(csv_row[col_index], na_value, inferred_types[j])
    inferred_types = [t or dtypes.string for t in inferred_types]
    return [constant_op.constant([0 if t is not dtypes.string else ''], dtype=t) for t in inferred_types]