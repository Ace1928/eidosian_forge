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
def _infer_column_names(filenames, field_delim, use_quote_delim, file_io_fn):
    """Infers column names from first rows of files."""
    csv_kwargs = {'delimiter': field_delim, 'quoting': csv.QUOTE_MINIMAL if use_quote_delim else csv.QUOTE_NONE}
    with file_io_fn(filenames[0]) as f:
        try:
            column_names = next(csv.reader(f, **csv_kwargs))
        except StopIteration:
            raise ValueError(f'Failed when reading the header line of {filenames[0]}. Is it an empty file?')
    for name in filenames[1:]:
        with file_io_fn(name) as f:
            try:
                if next(csv.reader(f, **csv_kwargs)) != column_names:
                    raise ValueError(f'All input CSV files should have the same column names in the header row. File {name} has different column names.')
            except StopIteration:
                raise ValueError(f'Failed when reading the header line of {name}. Is it an empty file?')
    return column_names