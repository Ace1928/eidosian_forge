import copy
import math
from typing import Sequence
import weakref
import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.distribute.ShardedVariable', v1=[])
class ShardedVariable(ShardedVariableMixin, composite_tensor.CompositeTensor):
    """A container for `Variables` that should be treated as shards.

  Variables that are too large to fit on a single device (e.g., large
  embeddings)
  may need to be sharded over multiple devices. This class maintains a list of
  smaller variables that can be independently stored on separate devices (eg,
  multiple parameter servers), and saves and restores those variables as if they
  were a single larger variable.

  Objects of this class can be saved with a given number of shards and then
  restored from a checkpoint into a different number of shards.

  Objects of this class can be saved to SavedModel format using
  `tf.saved_model.save`. The SavedModel can be used by programs like TF serving
  APIs. It is not yet supported to load the SavedModel with
  `tf.saved_model.load`.

  Since `ShardedVariable` can be saved and then restored to different number of
  shards depending on the restore environments, for example, TF serving APIs
  would restore to one shard for serving efficiency, when using
  `ShardedVariable` in a tf.function, one should generally not assume it has the
  same number of shards across save and load.

  Sharding is only supported along the first dimension.

  >>> class Model(tf.Module):
  ...   def __init__(self):
  ...     self.sharded_variable = ShardedVariable([
  ...       tf.Variable([3.0], dtype=tf.float32),
  ...       tf.Variable([2.0], dtype=tf.float32)
  ...     ])
  ...
  ...   @tf.function(input_signature=[tf.TensorSpec([], dtype=tf.int32)])
  ...   def fn(self, x):
  ...     return tf.nn.embedding_lookup(self.sharded_variable.variables, x)
  ...
  ...   @tf.function(input_signature=[tf.TensorSpec([], dtype=tf.int32)])
  ...   def serve_fn(self, x):
  ...     return tf.nn.embedding_lookup(self.sharded_variable.variables, x)
  >>>
  >>> model = Model()
  >>> model.fn(1).numpy()
  2.0
  >>> tf.saved_model.save(model, export_dir='/tmp/saved_model',
  ...   signatures=model.serve_fn)
  """

    @property
    def _type_spec(self):
        return ShardedVariableSpec(*(resource_variable_ops.VariableSpec(v.shape, v.dtype) for v in self._variables))

    @classmethod
    def _overload_all_operators(cls):
        """Register overloads for all operators."""
        for operator in tensor_lib.Tensor.OVERLOADABLE_OPERATORS:
            if operator == '__getitem__':
                continue
            cls._overload_operator(operator)

    @classmethod
    def _overload_operator(cls, operator):
        """Delegate an operator overload to `tensor_lib.Tensor`."""
        tensor_operator = getattr(tensor_lib.Tensor, operator)

        def _operator(v, *args, **kwargs):
            return tensor_operator(_var_to_tensor(v), *args, **kwargs)
        setattr(cls, operator, _operator)

    def __tf_experimental_restore_capture__(self, concrete_function, internal_capture):
        return None

    def _should_act_as_resource_variable(self):
        """Pass resource_variable_ops.is_resource_variable check."""
        return True

    def _write_object_proto(self, proto, options):
        resource_variable_ops.write_object_proto_for_resource_variable(self._saving_variable, proto, options, enforce_naming=False)