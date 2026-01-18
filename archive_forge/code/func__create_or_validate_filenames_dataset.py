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
def _create_or_validate_filenames_dataset(filenames, name=None):
    """Creates (or validates) a dataset of filenames.

  Args:
    filenames: Either a list or dataset of filenames. If it is a list, it is
      convert to a dataset. If it is a dataset, its type and shape is validated.
    name: (Optional.) A name for the tf.data operation.

  Returns:
    A dataset of filenames.
  """
    if isinstance(filenames, data_types.DatasetV2):
        element_type = dataset_ops.get_legacy_output_types(filenames)
        if element_type != dtypes.string:
            raise TypeError(f'The `filenames` argument must contain `tf.string` elements. Got a dataset of `{element_type!r}` elements.')
        element_shape = dataset_ops.get_legacy_output_shapes(filenames)
        if not element_shape.is_compatible_with(tensor_shape.TensorShape([])):
            raise TypeError(f'The `filenames` argument must contain `tf.string` elements of shape [] (i.e. scalars). Got a dataset of element shape {element_shape!r}.')
    else:
        filenames = nest.map_structure(_normalise_fspath, filenames)
        filenames = ops.convert_to_tensor(filenames, dtype_hint=dtypes.string)
        if filenames.dtype != dtypes.string:
            raise TypeError(f'The `filenames` argument must contain `tf.string` elements. Got `{filenames.dtype!r}` elements.')
        filenames = array_ops.reshape(filenames, [-1], name='flat_filenames')
        filenames = from_tensor_slices_op._TensorSliceDataset(filenames, is_files=True, name=name)
    return filenames