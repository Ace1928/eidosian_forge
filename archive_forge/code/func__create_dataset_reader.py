import os
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import from_tensor_slices_op
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _create_dataset_reader(dataset_creator, filenames, num_parallel_reads=None, name=None):
    """Creates a dataset that reads the given files using the given reader.

  Args:
    dataset_creator: A function that takes in a single file name and returns a
      dataset.
    filenames: A `tf.data.Dataset` containing one or more filenames.
    num_parallel_reads: The number of parallel reads we should do.
    name: (Optional.) A name for the tf.data operation.

  Returns:
    A `Dataset` that reads data from `filenames`.
  """

    def read_one_file(filename):
        filename = ops.convert_to_tensor(filename, dtypes.string, name='filename')
        return dataset_creator(filename)
    if num_parallel_reads is None:
        return filenames.flat_map(read_one_file, name=name)
    elif num_parallel_reads == dataset_ops.AUTOTUNE:
        return filenames.interleave(read_one_file, num_parallel_calls=num_parallel_reads, name=name)
    else:
        return ParallelInterleaveDataset(filenames, read_one_file, cycle_length=num_parallel_reads, block_length=1, sloppy=False, buffer_output_elements=None, prefetch_input_elements=None, name=name)