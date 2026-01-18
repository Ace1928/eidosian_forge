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
def _experimental_distribute_dataset(self, dataset, options):
    batch_size = distribute.compute_batch_size(dataset)
    if batch_size.numpy() < 0:
        raise ValueError('DTensor strategy requires a static batch size for now.The dynamic batch size will be supported in future')
    dataset = dataset.unbatch()

    def _create_batch_layout(tensor_spec):
        rank = len(tensor_spec.shape) + 1
        return layout.Layout.batch_sharded(self._mesh, batch_dim=dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, rank=rank)
    layouts = nest.map_structure(_create_batch_layout, dataset.element_spec)
    return input_util.DTensorDataset(dataset=dataset, mesh=self._mesh, layouts=layouts, global_batch_size=batch_size, dataset_already_batched=False, batch_dim=dtensor_util.DEFAULT_BATCH_MESH_DIM_NAME, prefetch=None, tf_data_service_config=None)