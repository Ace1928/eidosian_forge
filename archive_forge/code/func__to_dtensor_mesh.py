import tensorflow as tf
from tensorflow.experimental import dtensor
def _to_dtensor_mesh(device_mesh):
    """Convert the DeviceMesh to Tensorflow backend specific Mesh.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A `tf.dtensor.Mesh` instance.
    """
    mesh_dims = list(zip(device_mesh.axis_names, device_mesh.shape))
    return dtensor.create_distributed_mesh(mesh_dims=mesh_dims, local_devices=device_mesh.devices.flatten())