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
class MultiDeviceIterator:
    """An iterator over multiple devices."""

    def __init__(self, dataset, devices, max_buffer_size=1, prefetch_buffer_size=1, source_device='/cpu:0'):
        """Constructs a MultiDeviceIterator.

    Args:
      dataset: The input dataset to be iterated over.
      devices: The list of devices to fetch data to.
      max_buffer_size: Maximum size of the host side per device buffer to keep.
      prefetch_buffer_size: if > 0, then we setup a buffer on each device to
        prefetch into.
      source_device: The host device to place the `dataset` on.  In order to
        prevent deadlocks, if the prefetch_buffer_size is greater than the
        max_buffer_size, we set the max_buffer_size to prefetch_buffer_size.
    """
        options = options_lib.Options()
        options.experimental_distribute.num_devices = len(devices)
        if prefetch_buffer_size == 0:
            options.experimental_optimization.inject_prefetch = False
        dataset = dataset.with_options(options)
        self._dataset = dataset._apply_debug_options()
        self._experimental_slack = dataset.options().experimental_slack
        self._devices = devices
        self._source_device = source_device
        self._source_device_tensor = ops.convert_to_tensor(source_device)
        self._max_buffer_size = max_buffer_size
        self._prefetch_buffer_size = prefetch_buffer_size
        if self._prefetch_buffer_size > self._max_buffer_size:
            self._max_buffer_size = self._prefetch_buffer_size
        with ops.device(self._source_device):
            shared_name = ''
            if context.executing_eagerly():
                shared_name = context.anonymous_name()
            self._multi_device_iterator_resource = gen_dataset_ops.multi_device_iterator(devices=self._devices, shared_name=shared_name, container='', **self._dataset._flat_structure)
            if context.executing_eagerly():
                self._resource_deleter = resource_variable_ops.EagerResourceDeleter(handle=self._multi_device_iterator_resource, handle_device=self._source_device)
            self._incarnation_id = gen_dataset_ops.multi_device_iterator_init(self._dataset._variant_tensor, self._multi_device_iterator_resource, max_buffer_size=self._max_buffer_size)
        self._prototype_device_datasets = []
        for i, device in enumerate(self._devices):
            with ops.device(device):
                ds = _PerDeviceGenerator(i, self._multi_device_iterator_resource, self._incarnation_id, self._source_device_tensor, self._dataset.element_spec, iterator_is_anonymous=False)
                self._prototype_device_datasets.append(ds)
        self._device_iterators = []
        for i, device in enumerate(self._devices):
            with ops.device(device):
                ds = _create_device_dataset(self._prototype_device_datasets[i], self._incarnation_id, self._prefetch_buffer_size, self._experimental_slack)
                if context.executing_eagerly():
                    self._device_iterators.append(dataset_ops.make_one_shot_iterator(ds))
                else:
                    self._device_iterators.append(dataset_ops.make_initializable_iterator(ds))
        if not context.executing_eagerly():
            device_iterator_initializers = [iterator.initializer for iterator in self._device_iterators]
            self._initializer = control_flow_ops.group(*device_iterator_initializers)

    def get_next(self, device=None):
        """Returns the next element given a `device`, else returns all in a list."""
        if device is not None:
            index = self._devices.index(device)
            return self._device_iterators[index].get_next()
        result = []
        for i, device in enumerate(self._devices):
            with ops.device(device):
                result.append(self._device_iterators[i].get_next())
        return result

    def get_next_as_optional(self):
        result = []
        for i, device in enumerate(self._devices):
            with ops.device(device):
                result.append(self._device_iterators[i].get_next_as_optional())
        return result

    @property
    def initializer(self):
        if context.executing_eagerly():
            return control_flow_ops.no_op()
        return self._initializer

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

    @property
    def element_spec(self):
        return self._dataset.element_spec