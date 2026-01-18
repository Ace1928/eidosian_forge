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
def MapFn(unused_input):
    source_dataset_output_types = dataset_ops.get_legacy_output_types(source_dataset)
    if isinstance(source_dataset_output_types, dtypes.DType):
        output_types = [source_dataset_output_types]
    elif isinstance(source_dataset_output_types, (list, tuple)):
        output_types = source_dataset_output_types
    else:
        raise ValueError('Source dataset has invalid output types. Only list/tuples or TensorFlow tensor types are accepted.')
    remote_calls = functional_ops.remote_call(args=[source_handle], Tout=output_types, f=LoadingFunc, target='/job:%s/replica:0/task:0/cpu:0' % file_reader_job)
    if len(remote_calls) == 1:
        return remote_calls[0]
    else:
        return remote_calls