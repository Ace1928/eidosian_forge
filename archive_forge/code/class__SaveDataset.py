import os
from tensorflow.python.checkpoint import checkpoint as checkpoint_lib
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import nested_structure_coder
class _SaveDataset(dataset_ops.UnaryDataset):
    """"A dataset that loads previously saved dataset."""

    def __init__(self, dataset, path, shard_func, compression):
        self._element_spec = dataset.element_spec
        self._shard_func = shard_func
        dataset, shard_func, use_shard_func, path = set_save_dataset_attributes(dataset, shard_func, path)
        variant_tensor = ged_ops.save_dataset_v2(dataset._variant_tensor, path=path, shard_func_other_args=shard_func.captured_inputs, shard_func=shard_func, use_shard_func=use_shard_func, compression=compression, output_types=structure.get_flat_tensor_types(dataset.element_spec), output_shapes=structure.get_flat_tensor_shapes(dataset.element_spec))
        super().__init__(dataset, variant_tensor)

    def _functions(self):
        return [self._shard_func]

    @property
    def element_spec(self):
        return self._element_spec