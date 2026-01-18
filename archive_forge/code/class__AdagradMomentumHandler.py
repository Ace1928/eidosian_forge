import collections
import copy
import math
import re
from typing import Optional
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export
class _AdagradMomentumHandler(_OptimizerHandler):
    """Handles Adagrad with Momentum specific logic.

  Creates slot variables and defines their initializers. Defines load/retrieve
  operations to be used for loading variables into TPU memory (from host memory)
  and retrieving variables from TPU memory (into host memory).
  """

    def set_optimization_parameters(self, table_descriptor):
        table_descriptor.optimization_parameters.adagrad_momentum.SetInParent()
        table_descriptor.optimization_parameters.adagrad_momentum.momentum = self._optimization_parameters.momentum
        table_descriptor.optimization_parameters.adagrad_momentum.use_nesterov = self._optimization_parameters.use_nesterov
        table_descriptor.optimization_parameters.adagrad_momentum.exponent = self._optimization_parameters.exponent
        table_descriptor.optimization_parameters.adagrad_momentum.beta2 = self._optimization_parameters.beta2
        table_descriptor.optimization_parameters.adagrad_momentum.epsilon = self._optimization_parameters.epsilon

    def get_default_slot_variable_names(self, table):
        return AdagradMomentumSlotVariableNames('{}/{}/Accumulator'.format(table, 'AdagradMomentum'), '{}/{}/Momentum'.format(table, 'AdagradMomentum'))

    def create_variables_and_ops(self, table, slot_variable_names, num_hosts, table_config, table_variables, config_proto):
        accumulator_initializer = init_ops.zeros_initializer()
        accumulator_variables = _create_partitioned_variables(name=slot_variable_names.accumulator, num_hosts=num_hosts, vocabulary_size=table_config.vocabulary_size, embedding_dimension=table_config.dimension, collections=[ops.GraphKeys.GLOBAL_VARIABLES], initializer=accumulator_initializer)
        momenta_initializer = init_ops.zeros_initializer()
        momenta_variables = _create_partitioned_variables(name=slot_variable_names.momenta, num_hosts=num_hosts, vocabulary_size=table_config.vocabulary_size, embedding_dimension=table_config.dimension, collections=[ops.GraphKeys.GLOBAL_VARIABLES], initializer=momenta_initializer)
        slot_variables = AdagradMomentumSlotVariables(accumulator_variables, momenta_variables)

        def load_ops_fn():
            """Returns the load ops for AdaGrad with momentum embedding tables.

      Returns:
        A list of ops to load embedding and slot variables from CPU to TPU.
      """
            config = config_proto
            load_op_list = []
            for host_id, table_variable, accumulator_variable, momenta_variable in zip(range(num_hosts), table_variables, accumulator_variables, momenta_variables):
                with ops.colocate_with(table_variable):
                    load_parameters_op = tpu_ops.load_tpu_embedding_adagrad_momentum_parameters(parameters=table_variable, accumulators=accumulator_variable, momenta=momenta_variable, table_name=table, num_shards=num_hosts, shard_id=host_id, config=config)
                config = None
                load_op_list.append(load_parameters_op)
            return load_op_list

        def retrieve_ops_fn():
            """Returns the retrieve ops for AdaGrad with momentum embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
            config = config_proto
            retrieve_op_list = []
            for host_id, table_variable, accumulator_variable, momenta_variable in zip(range(num_hosts), table_variables, accumulator_variables, momenta_variables):
                with ops.colocate_with(table_variable):
                    retrieved_table, retrieved_accumulator, retrieved_momenta = tpu_ops.retrieve_tpu_embedding_adagrad_momentum_parameters(table_name=table, num_shards=num_hosts, shard_id=host_id, config=config)
                    retrieve_parameters_op = control_flow_ops.group(state_ops.assign(table_variable, retrieved_table), state_ops.assign(accumulator_variable, retrieved_accumulator), state_ops.assign(momenta_variable, retrieved_momenta))
                config = None
                retrieve_op_list.append(retrieve_parameters_op)
            return retrieve_op_list
        return (slot_variables, load_ops_fn, retrieve_ops_fn)