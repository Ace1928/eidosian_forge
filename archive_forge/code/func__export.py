from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.training import saver as saver_module
def _export(self):
    return gen_lookup_ops.lookup_table_export_v2(self.table_ref, dtypes.string, dtypes.float32)