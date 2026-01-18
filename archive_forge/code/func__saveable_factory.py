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
def _saveable_factory(name=self.name):
    """Creates `SaveableObject`s for this `ShardedVariable`."""
    saveables = []
    dims = len(self._variables[0].shape)
    var_offset = [0 for _ in range(dims)]
    for v in self._variables:
        save_slice_info = variables_lib.Variable.SaveSliceInfo(full_name=self.name, full_shape=self.shape.as_list(), var_offset=copy.copy(var_offset), var_shape=v.shape.as_list())
        saveables.append(saveable_object_util.ResourceVariableSaveable(v, save_slice_info.spec, name))
        var_offset[0] += int(v.shape[0])
    return saveables