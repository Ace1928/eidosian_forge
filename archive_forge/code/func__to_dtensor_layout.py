import tensorflow as tf
from tensorflow.experimental import dtensor
def _to_dtensor_layout(tensor_layout):
    """Convert the TensorLayout to Tensorflow backend specific Sharding.

    Args:
        tensor_layout: TensorLayout instance to convert.

    Returns:
        A `tf.dtensor.Layout` instance.
    """
    if tensor_layout.device_mesh is None:
        raise ValueError('Cannot create sharding when device mesh is not set for TensorLayout.')
    sharding_specs = [axis if axis else dtensor.UNSHARDED for axis in tensor_layout.axes]
    dtensor_mesh = _to_dtensor_mesh(tensor_layout.device_mesh)
    return dtensor.Layout(sharding_specs=sharding_specs, mesh=dtensor_mesh)