from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import prefetch_op
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import resource_variable_ops
def _eager_reset(self):
    """Resets the MultiDeviceIterator in eager mode."""
    if not ops.executing_eagerly_outside_functions():
        raise ValueError('Resetting a multi-device iterator is only supported in the eager mode.')
    self._incarnation_id = gen_dataset_ops.multi_device_iterator_init(self._dataset._variant_tensor, self._multi_device_iterator_resource, max_buffer_size=self._max_buffer_size)
    for i, device in enumerate(self._devices):
        with ops.device(device):
            ds = _create_device_dataset(self._prototype_device_datasets[i], self._incarnation_id, self._prefetch_buffer_size, self._experimental_slack)
            ds_variant = ds._variant_tensor
            gen_dataset_ops.make_iterator(ds_variant, self._device_iterators[i]._iterator_resource)