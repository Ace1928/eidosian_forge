from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.data.util import traverse
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.types import data as data_types
from tensorflow.python.types import distribute as distribute_types
def _clone_dataset(dataset):
    """Returns a cloned version of `dataset`."""
    variant_tensor_ops = traverse.obtain_all_variant_tensor_ops(dataset)
    remap_dict = _clone_helper(dataset._variant_tensor.op, variant_tensor_ops)
    new_variant_tensor = remap_dict[dataset._variant_tensor.op].outputs[0]
    return dataset_ops._VariantDataset(new_variant_tensor, dataset.element_spec)