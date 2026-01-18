from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
Uncompress a compressed dataset element.

  Args:
    element: A scalar variant tensor to uncompress. The element should have been
      created by calling `compress`.
    output_spec: A nested structure of `tf.TypeSpec` representing the type(s) of
      the uncompressed element.

  Returns:
    The uncompressed element.
  