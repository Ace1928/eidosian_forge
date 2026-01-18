import functools
import sys
import time
import six
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.distribute_lib import InputReplicationMode
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import distribute as distribute_types
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
class _SingleWorkerOwnedDatasetIterator(_SingleWorkerDatasetIteratorBase, composite_tensor.CompositeTensor):
    """Iterator for a DistributedDataset instance."""

    def __init__(self, dataset=None, worker=None, devices=None, components=None, element_spec=None, options=None, canonicalize_devices=None):
        """Create iterator for the `dataset` to fetch data to worker's `devices` .

    `OwnedMultiDeviceIterator` is used to prefetch input to the devices on the
    given worker. The lifetime of this iterator is tied to the encompassing
    python object. Once we go out of scope of the python object or return from
    a tf.function the underlying iterator resource is deleted.

    Args:
      dataset: A `tf.data.Dataset` instance.
      worker: Worker on which ops should be created.
      devices: Distribute data from `dataset` to these devices.
      components: Tensor components to construct the
        _SingleWorkerOwnedDatasetIterator from.
      element_spec: A nested structure of `TypeSpec` objects that represents the
      type specification of elements of the iterator.
      options: `tf.distribute.InputOptions` used to control options on how this
      dataset is distributed.
      canonicalize_devices: Whether to canonicalize devices for workers fully or
      partially. If False, it will partially canonicalize devices by removing
      job and task.
    """
        if worker is None or devices is None:
            raise ValueError('Both `worker` and `devices` should be provided')
        error_message = 'Either `dataset` or both `components` and `element_spec` need to be provided.'
        self._options = options
        self._canonicalize_devices = canonicalize_devices
        if dataset is None:
            if components is None or element_spec is None:
                raise ValueError(error_message)
            self._element_spec = element_spec
            self._worker = worker
            self._devices = devices
            self._iterator = components[0]
        else:
            if components is not None or element_spec is not None:
                raise ValueError(error_message)
            super(_SingleWorkerOwnedDatasetIterator, self).__init__(dataset, worker, devices, self._options)

    def _create_owned_multi_device_iterator(self):
        if not ops.inside_function():
            device_scope = device_util.canonicalize(self._worker, device_util.current())
            host_device = device_util.get_host_for_device(device_scope)
        else:
            device_scope, host_device = (self._worker, self._worker)
        with ops.device(device_scope):
            if self._options is not None:
                self._iterator = multi_device_iterator_ops.OwnedMultiDeviceIterator(self._dataset, self._devices, source_device=host_device, max_buffer_size=self._options.experimental_per_replica_buffer_size, prefetch_buffer_size=self._options.experimental_per_replica_buffer_size)
            else:
                self._iterator = multi_device_iterator_ops.OwnedMultiDeviceIterator(self._dataset, self._devices, source_device=host_device)

    def _make_iterator(self):
        """Make appropriate iterator on the dataset."""
        if not self._worker:
            raise ValueError('Worker device must be specified when creating an owned iterator.')
        if _should_use_multi_device_iterator(self._options):
            self._create_owned_multi_device_iterator()
        else:
            with ops.device(self._worker):
                self._iterator = iter(self._dataset)

    @property
    def element_spec(self):
        return self._element_spec

    @property
    def _type_spec(self):
        return _SingleWorkerDatasetIteratorSpec(self._worker, self._devices, self._element_spec, self._options, self._canonicalize_devices)

    @property
    def output_classes(self):
        """Returns the class of each component of an element of this iterator.

    The expected values are `tf.Tensor` and `tf.SparseTensor`.

    Returns:
      A nested structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_classes(), self._element_spec)

    @property
    def output_shapes(self):
        """Returns the shape of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_shapes(), self._element_spec)

    @property
    def output_types(self):
        """Returns the type of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
        return nest.map_structure(lambda component_spec: component_spec._to_legacy_output_types(), self._element_spec)