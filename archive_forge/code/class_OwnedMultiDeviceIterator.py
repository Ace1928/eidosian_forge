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
class OwnedMultiDeviceIterator(composite_tensor.CompositeTensor):
    """An iterator over multiple devices.

  The multi-device iterator resource created through `OwnedMultiDeviceIterator`
  is owned by the Python object and the life time of the underlying resource is
  tied to the life time of the `OwnedMultiDeviceIterator` object. This makes
  `OwnedMultiDeviceIterator` appropriate for use in eager mode and inside of
  tf.functions.
  """

    def __init__(self, dataset=None, devices=None, max_buffer_size=1, prefetch_buffer_size=1, source_device='/cpu:0', components=None, element_spec=None):
        """Constructs an owned MultiDeviceIterator object.

    Args:
      dataset: The input dataset to be iterated over.
      devices: (Required.) The list of devices to fetch data to.
      max_buffer_size: Maximum size of the host side per device buffer to keep.
      prefetch_buffer_size: if > 0, then we setup a buffer on each device to
        prefetch into.
      source_device: The host device to place the `dataset` on.  In order to
        prevent deadlocks, if the prefetch_buffer_size is greater than the
        max_buffer_size, we set the max_buffer_size to prefetch_buffer_size.
      components: Tensor components to construct the MultiDeviceIterator from.
      element_spec: A (nested) structure of `tf.TypeSpec` objects that
        represents the type specification of elements of the iterator.

    Raises:
      RuntimeError: If executed in graph mode or outside of function building
        mode.
      ValueError: If any of the following happens:
        - `devices` is `None`
        - `dataset` is `None` and either `components` or `element_spec` is
          `None`
        - `dataset` is not None and either `components` or `element_spec` is
          provided
    """
        if not context.executing_eagerly() and (not ops.inside_function()):
            raise RuntimeError('OwnedMultiDeviceIterator is only supported inside of tf.function or when eager execution is enabled.')
        if devices is None:
            raise ValueError('`devices` must be provided.')
        if dataset is None:
            if components is None or element_spec is None:
                raise ValueError('When `dataset` is not provided, both `components` and `element_spec` must be specified.')
            self._element_spec = element_spec
            self._devices = devices
            self._source_device = source_device
            self._multi_device_iterator_resource = components[0]
            self._device_iterators = components[1:]
        else:
            if components is not None or element_spec is not None:
                raise ValueError('When `dataset` is provided, `element_spec` and `components` must not be specified.')
            options = options_lib.Options()
            options.experimental_distribute.num_devices = len(devices)
            if prefetch_buffer_size == 0:
                options.experimental_optimization.inject_prefetch = False
            dataset = dataset.with_options(options)
            dataset = dataset._apply_debug_options()
            self._element_spec = dataset.element_spec
            experimental_slack = dataset.options().experimental_slack
            self._devices = devices
            self._source_device = source_device
            source_device_tensor = ops.convert_to_tensor(self._source_device)
            if prefetch_buffer_size > max_buffer_size:
                max_buffer_size = prefetch_buffer_size
            with ops.device(self._source_device):
                self._multi_device_iterator_resource = gen_dataset_ops.anonymous_multi_device_iterator_v3(devices=self._devices, **dataset._flat_structure)
                incarnation_id = gen_dataset_ops.multi_device_iterator_init(dataset._variant_tensor, self._multi_device_iterator_resource, max_buffer_size=max_buffer_size)
            prototype_device_datasets = []
            for i, device in enumerate(self._devices):
                with ops.device(device):
                    ds = _PerDeviceGenerator(i, self._multi_device_iterator_resource, incarnation_id, source_device_tensor, dataset.element_spec, iterator_is_anonymous=True)
                    prototype_device_datasets.append(ds)
            self._device_iterators = []
            for i, device in enumerate(self._devices):
                with ops.device(device):
                    ds = _create_device_dataset(prototype_device_datasets[i], incarnation_id, prefetch_buffer_size, experimental_slack)
                    iterator = iter(ds)
                    self._device_iterators.append(iterator)

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

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        try:
            return self.get_next()
        except errors.OutOfRangeError:
            raise StopIteration

    def get_next_as_optional(self):
        result = []
        for i, device in enumerate(self._devices):
            with ops.device(device):
                result.append(self._device_iterators[i].get_next_as_optional())
        return result

    @property
    def element_spec(self):
        return self._element_spec

    @property
    def _type_spec(self):
        return MultiDeviceIteratorSpec(self._devices, self._source_device, self._element_spec)