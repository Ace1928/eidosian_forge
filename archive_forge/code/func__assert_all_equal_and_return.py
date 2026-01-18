import collections
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import serialization
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _assert_all_equal_and_return(tensors, name=None):
    """Asserts that all tensors are equal and returns the first one."""
    with ops.name_scope(name, 'assert_all_equal', values=tensors):
        if len(tensors) == 1:
            return tensors[0]
        assert_equal_ops = []
        for t in tensors[1:]:
            assert_equal_ops.append(check_ops.assert_equal(tensors[0], t))
        with ops.control_dependencies(assert_equal_ops):
            return array_ops.identity(tensors[0])