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
def _decompose_indexed_slices(self, indexed_slices):
    """Decompose a global `IndexedSlices` into a list of per-variable ones."""
    per_var_indices, partition_assignments = self._decompose_indices(indexed_slices.indices)
    per_var_values = data_flow_ops.dynamic_partition(indexed_slices.values, partition_assignments, len(self._variables))
    return [indexed_slices_lib.IndexedSlices(values=per_var_values[i], indices=per_var_indices[i]) for i in range(len(self._variables))]