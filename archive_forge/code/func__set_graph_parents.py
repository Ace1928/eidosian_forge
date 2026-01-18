import abc
import contextlib
import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.linalg import slicing
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _set_graph_parents(self, graph_parents):
    """Set self._graph_parents.  Called during derived class init.

    This method allows derived classes to set graph_parents, without triggering
    a deprecation warning (which is invoked if `graph_parents` is passed during
    `__init__`.

    Args:
      graph_parents: Iterable over Tensors.
    """
    graph_parents = [] if graph_parents is None else graph_parents
    for i, t in enumerate(graph_parents):
        if t is None or not (linear_operator_util.is_ref(t) or tensor_util.is_tf_type(t)):
            raise ValueError('Graph parent item %d is not a Tensor; %s.' % (i, t))
    self._graph_parents = graph_parents