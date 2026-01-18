import functools
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def _create_batch_layout(tensor_spec):
    rank = len(tensor_spec.shape)
    return layout.Layout.batch_sharded(self._mesh, batch_dim=dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, rank=rank)