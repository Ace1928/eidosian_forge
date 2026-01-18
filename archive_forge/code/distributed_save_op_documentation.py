from tensorflow.core.protobuf import snapshot_pb2
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.saved_model import nested_structure_coder
Initiates the process of distributedly saving a dataset to disk.

  Args:
    dataset: The `tf.data.Dataset` to save.
    path: A string indicating the filepath of the directory to which to save
      `dataset`.
    dispatcher_address: A string indicating the address of the dispatcher for
      the tf.data service instance used to save `dataset`.
    compression: (Optional.) A string indicating whether and how to compress the
      `dataset` materialization.  If `"AUTO"`, the tf.data runtime decides which
      algorithm to use.  If `"GZIP"` or `"SNAPPY"`, that specific algorithm is
      used.  If `None`, the `dataset` materialization is not compressed.

  Returns:
    An operation which when executed performs the distributed save.

  Raises:
    ValueError: If `dispatcher_address` is invalid.
  