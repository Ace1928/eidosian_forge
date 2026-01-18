import os
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.experimental import dtensor_strategy_extended
from tensorflow.python.distribute.experimental import dtensor_util
def _build_distributed_mesh(batch_dim_name):
    device_type = d_config.preferred_device_type()
    local_devices = d_config.local_devices(device_type)
    number_clients = d_config.num_clients()
    dtensor_util.initialize_accelerator_system_once(device_type)
    mesh_dims = [(batch_dim_name, len(local_devices) * number_clients)]
    return mesh_util.create_distributed_mesh(mesh_dims, device_type=device_type)