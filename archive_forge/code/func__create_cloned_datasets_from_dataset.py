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
def _create_cloned_datasets_from_dataset(self, dataset, input_context, input_workers, strategy, num_replicas_in_sync):
    if num_replicas_in_sync is not None and num_replicas_in_sync > 1:
        num_workers = input_context.num_input_pipelines if input_context else len(input_workers.worker_devices)
        rebatch_fn = self._make_rebatch_fn(dataset, num_workers, num_replicas_in_sync)
    else:
        rebatch_fn = None
    self._cloned_datasets = []
    if input_context:
        assert input_workers.num_workers == 1
        if rebatch_fn is not None:
            dataset = rebatch_fn(dataset, input_context.input_pipeline_id)
        dataset = input_ops.auto_shard_dataset(dataset, input_context.num_input_pipelines, input_context.input_pipeline_id, num_replicas_in_sync)
        self._cloned_datasets.append(dataset)
    else:
        replicated_ds = distribute.replicate(dataset, input_workers.worker_devices)
        for i, worker in enumerate(input_workers.worker_devices):
            with ops.device(worker):
                cloned_dataset = replicated_ds[worker]
                if rebatch_fn is not None:
                    cloned_dataset = rebatch_fn(cloned_dataset, i)
                cloned_dataset = input_ops.auto_shard_dataset(cloned_dataset, len(input_workers.worker_devices), i, num_replicas_in_sync)
                self._cloned_datasets.append(cloned_dataset)