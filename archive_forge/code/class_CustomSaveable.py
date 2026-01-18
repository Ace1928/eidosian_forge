from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.training import saver as saver_module
class CustomSaveable(saver_module.BaseSaverBuilder.SaveableObject):
    """A custom saveable for CheckpointedOp."""

    def __init__(self, table, name):
        tensors = table._export()
        specs = [saver_module.BaseSaverBuilder.SaveSpec(tensors[0], '', name + '-keys'), saver_module.BaseSaverBuilder.SaveSpec(tensors[1], '', name + '-values')]
        super(CheckpointedOp.CustomSaveable, self).__init__(table, specs, name)

    def restore(self, restore_tensors, shapes):
        return gen_lookup_ops.lookup_table_import_v2(self.op.table_ref, restore_tensors[0], restore_tensors[1])