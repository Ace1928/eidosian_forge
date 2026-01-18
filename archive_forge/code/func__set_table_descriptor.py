import abc
import math
import typing
from typing import Any, Dict, Callable, Iterable, List, Optional, Text, Tuple, TypeVar, Union
from absl import logging
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import ops
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export
def _set_table_descriptor(self, table_descriptor: tpu_embedding_configuration_pb2.TPUEmbeddingConfiguration.TableDescriptor, num_hosts: int, learning_rate_index: Dict[Callable[[], Any], int]):
    """Set the table descriptor from the table data."""
    table_descriptor.name = self.name
    table_descriptor.vocabulary_size = max(self.vocabulary_size, num_hosts)
    table_descriptor.dimension = self.dim
    parameters = table_descriptor.optimization_parameters
    if self.optimizer:
        if callable(self.optimizer.learning_rate):
            parameters.learning_rate.dynamic.tag = learning_rate_index[self.optimizer.learning_rate]
        else:
            parameters.learning_rate.constant = self.optimizer.learning_rate
        if self.optimizer.low_dimensional_packing_status:
            parameters.low_dimensional_packing_status = optimization_parameters_pb2.LowDimensionalPackingStatus.Status.ENABLED
        self.optimizer._set_optimization_parameters(parameters)
    if self.quantization_config:
        self.quantization_config._set_optimization_parameters(parameters)