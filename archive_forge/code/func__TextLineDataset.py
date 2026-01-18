from typing import Callable, Optional, Text, Union
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.types import data as data_types
def _TextLineDataset(filename: Text) -> dataset_ops.Dataset:
    buffer_size = 8 * 1024 * 1024
    dataset = readers.TextLineDataset(filename, buffer_size=buffer_size)
    return dataset