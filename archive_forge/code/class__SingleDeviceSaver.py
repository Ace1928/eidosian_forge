from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
class _SingleDeviceSaver(object):
    """Saves and restores checkpoints from the current device."""
    __slots__ = ['_tensor_slice_dict']

    def __init__(self, tensor_slice_dict):
        """Specify a list of `SaveableObject`s to save and restore.

    Args:
      tensor_slice_dict: A dict mapping checkpoint key -> slice_spec -> tensor.
    """
        self._tensor_slice_dict = tensor_slice_dict

    def save(self, file_prefix, options=None):
        """Save the saveable objects to a checkpoint with `file_prefix`.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix to
        save under.
      options: Optional `CheckpointOptions` object.
    Returns:
      An `Operation`, or None when executing eagerly.
    """
        options = options or checkpoint_options.CheckpointOptions()
        tensor_names = []
        tensors = []
        slice_specs = []
        for checkpoint_key, tensor_slices in self._tensor_slice_dict.items():
            for slice_spec, tensor in tensor_slices.items():
                if isinstance(tensor, saveable_object.SaveSpec):
                    tensor_value = tensor.tensor
                    if tensor_value is not None:
                        tensor_names.append(tensor.name)
                        tensors.append(tensor_value)
                        slice_specs.append(tensor.slice_spec)
                else:
                    tensor_names.append(checkpoint_key)
                    tensors.append(tensor)
                    slice_specs.append(slice_spec)
        save_device = options.experimental_io_device or (len(tensors) and saveable_object_util.set_cpu0(tensors[0].device))
        save_device = save_device or 'cpu:0'
        with ops.device(save_device):
            return io_ops.save_v2(file_prefix, tensor_names, slice_specs, tensors)

    def restore(self, file_prefix, options=None):
        """Restore the saveable objects from a checkpoint with `file_prefix`.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix for
        files to read from.
      options: Optional `CheckpointOptions` object.

    Returns:
      A restored tensor dict (maps checkpoint_key -> slice_spec -> tensor).
    """
        options = options or checkpoint_options.CheckpointOptions()
        tensor_names = []
        tensor_dtypes = []
        slice_specs = []
        for checkpoint_key, tensor_slices in self._tensor_slice_dict.items():
            for slice_spec, tensor in tensor_slices.items():
                tensor_dtypes.append(tensor.dtype)
                if isinstance(tensor, saveable_object.SaveSpec):
                    slice_specs.append(tensor.slice_spec)
                    tensor_names.append(tensor.name)
                else:
                    slice_specs.append(slice_spec)
                    tensor_names.append(checkpoint_key)
        restore_device = options.experimental_io_device or 'cpu:0'
        with ops.device(restore_device):
            restored_tensors = io_ops.restore_v2(file_prefix, tensor_names, slice_specs, tensor_dtypes)
        restored_tensor_dict = {}
        for checkpoint_key, tensor_slices in self._tensor_slice_dict.items():
            for slice_spec in tensor_slices:
                restored_tensor = restored_tensors.pop(0)
                restored_tensor_dict.setdefault(checkpoint_key, {})[slice_spec] = restored_tensor
        return restored_tensor_dict