from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.deprecation import deprecated
class DistributedIteratorV1(input_lib.DistributedIteratorBase):
    """Input Iterator for a distributed dataset."""

    @property
    def _initializer(self):
        init_ops = []
        for it in self._iterators:
            init_ops.extend(it.initialize())
        return control_flow_ops.group(init_ops)

    @deprecated(None, "Use the iterator's `initializer` property instead.")
    def initialize(self):
        """Initialize underlying iterators.

    Returns:
      A list of any initializer ops that should be run.
    """
        return self._initializer

    @property
    def initializer(self):
        """Returns a list of ops that initialize the iterator."""
        return self.initialize()

    @property
    def output_classes(self):
        return self._iterators[0].output_classes

    @property
    def output_shapes(self):
        return self._iterators[0].output_shapes

    @property
    def output_types(self):
        return self._iterators[0].output_types

    def get_iterator(self, worker):
        for i, w in enumerate(self._input_workers.worker_devices):
            if worker == w:
                return self._iterators[i]
        return None

    @property
    def element_spec(self):
        """The type specification of an element of this iterator."""
        return self._element_spec