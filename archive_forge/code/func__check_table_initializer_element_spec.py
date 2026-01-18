from tensorflow.python.data.experimental.ops.cardinality import assert_cardinality
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
def _check_table_initializer_element_spec(element_spec):
    """Raises an error if the given table initializer element spec is invalid."""
    base_error = 'Datasets used to initialize lookup tables must produce elements in the form (key, value), where the keys and values are scalar tensors. '
    specific_error = None
    if len(element_spec) != 2:
        raise ValueError(base_error + f'However, the given dataset produces {len(element_spec)} components instead of two (key, value) components. Full dataset element spec: {element_spec}.')
    if not isinstance(element_spec[0], tensor.TensorSpec):
        raise ValueError(base_error + f'However, the given dataset produces non-Tensor keys of type {type(element_spec[0])}.')
    if not isinstance(element_spec[1], tensor.TensorSpec):
        raise ValueError(base_error + f'However, the given dataset produces non-Tensor values of type {type(element_spec[1])}.')
    if element_spec[0].shape.rank not in (None, 0):
        raise ValueError(base_error + f'However, the given dataset produces non-scalar key Tensors of rank {element_spec[0].shape.rank}.')
    if element_spec[1].shape.rank not in (None, 0):
        raise ValueError(base_error + f'However, the given dataset produces non-scalar value Tensors of rank {element_spec[1].shape.rank}.')