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
def _create_iterators_per_worker(worker_datasets, input_workers, options=None):
    """Create a multidevice iterator on each of the workers."""
    assert isinstance(input_workers, input_lib.InputWorkers)
    assert len(worker_datasets) == len(input_workers.worker_devices)
    iterators = []
    for i, worker in enumerate(input_workers.worker_devices):
        with ops.device(worker):
            worker_devices = input_workers.compute_devices_for_worker(i)
            iterator = _SingleWorkerDatasetIterator(worker_datasets[i], worker, worker_devices, options)
            iterators.append(iterator)
    return iterators