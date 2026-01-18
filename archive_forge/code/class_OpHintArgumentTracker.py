import collections as _collections
import copy as _copy
import json as _json
import uuid as _uuid
from tensorflow.core.framework import attr_value_pb2 as _attr_value_pb2
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.core.framework import node_def_pb2 as _node_def_pb2
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.framework.graph_util_impl import _bfs_for_reachable_nodes
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.util import compat as _compat
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export as _tf_export
class OpHintArgumentTracker:
    """Conceptually tracks indices of arguments of "OpHint functions".

    The inputs and arguments of these functions both use an instance
    of the class so they can have independent numbering.
    """

    def __init__(self, function_name, unique_function_id, node_name_prefix, attr_name, level=1, children_inputs_mappings=None):
        """Initialize ophint argument.

      Args:
        function_name: Name of the function that this tracks arguments for.
        unique_function_id: UUID of function that this tracks arguments for.
        node_name_prefix: How identities that are created are named.
        attr_name: Name of attribute to use to store the index for this hint.
          i.e. FUNCTION_INPUT_INDEX or FUNCTION_OUTPUT_INDEX
        level: Hierarchical level of the Ophint node, a number.
        children_inputs_mappings: Inputs/Outputs mapping for children hints.
      """
        self._function_name = function_name
        self._unique_function_id = unique_function_id
        self._next_global_index = 0
        self._used_global_indices = set()
        self._tag_to_global_index = {}
        self._tag_to_next_sort_index = {}
        self._node_name_prefix = node_name_prefix
        self._attr_name = attr_name
        self._level = level
        self._children_inputs_mappings = children_inputs_mappings

    def _get_new_global_index(self, index_override):
        """Return the next unused argument index in order or use an override.

      Args:
        index_override: An index to use instead of the next available or None
          to use the next available.

      Returns:
        A valid global_index to use for the next hint argument.

      Raises:
        ValueError: If the index_override is already used by another hint.
      """
        if index_override is None:
            global_index = self._next_global_index
        else:
            if index_override in self._used_global_indices:
                raise ValueError('Index %d was already used by another call to add')
            global_index = index_override
        self._used_global_indices.add(global_index)
        while self._next_global_index in self._used_global_indices:
            self._next_global_index += 1
        return global_index

    def add(self, arg, tag=None, name=None, aggregate=None, index_override=None):
        """Return a wrapped tensor of an input tensor as an argument.

      Args:
        arg: A TensorFlow tensor that should be considered an argument.
        tag: String tag to identify arguments that should be packed.
        name: Name of argument. This is included in the Identity hint op names.
        aggregate: Strategy to aggregate.
        Acceptable values are OpHint.AGGREGATE_FIRST, OpHint.AGGREGATE_LAST,
          and OpHint.AGGREGATE_STACK.
          Note, aggregate is only valid if tag is specified.
        index_override: Specify what input/output index should this be in the
          final stub. i.e. add(arg0, index=1); add(arg1, index=0) will make the
          final stub be as stub_func(inputs[arg1, arg0], outputs=[]) rather than
          the default call order based ordering.

      Returns:
        A tensor representing the wrapped argument.

      Raises:
        ValueError: When indices are not consistent.
      """
        if tag is None:
            if aggregate is not None:
                raise ValueError('You must specify `tag` if using aggregate.')
            global_index = self._get_new_global_index(index_override)
            sort_index = None
        else:
            if aggregate is None:
                raise ValueError('You must specify `aggregate` if using tag.')
            if tag not in self._tag_to_global_index:
                self._tag_to_global_index[tag] = self._get_new_global_index(index_override)
                self._tag_to_next_sort_index[tag] = 0
            elif index_override and index_override != self._tag_to_global_index[tag]:
                raise ValueError('Tag %r was called with two indices %r and %r' % (tag, index_override, self._tag_to_global_index[tag]))
            global_index = self._tag_to_global_index[tag]
            sort_index = self._tag_to_next_sort_index[tag]
            self._tag_to_next_sort_index[tag] += 1
        uuid = self._unique_function_id
        name = '%s-%s-%s-%r-%r-%s' % (self._node_name_prefix, self._function_name, uuid, global_index, sort_index, name)
        identity_op = _array_ops.identity(arg, name=name)
        identity_op.op._set_attr(OpHint.FUNCTION_NAME_ATTR, _attr_value_pb2.AttrValue(s=_compat.as_bytes(self._function_name)))
        identity_op.op._set_attr(OpHint.FUNCTION_UUID_ATTR, _attr_value_pb2.AttrValue(s=_compat.as_bytes(self._unique_function_id)))
        identity_op.op._set_attr(self._attr_name, _attr_value_pb2.AttrValue(i=global_index))
        identity_op.op._set_attr(OpHint.FUNCTION_LEVEL_ATTR, _attr_value_pb2.AttrValue(i=self._level))
        if self._children_inputs_mappings:
            identity_op.op._set_attr(OpHint.CHILDREN_INPUTS_MAPPINGS, _attr_value_pb2.AttrValue(s=_compat.as_bytes(_json.dumps(self._children_inputs_mappings))))
        if sort_index is not None:
            identity_op.op._set_attr(OpHint.FUNCTION_SORT_INDEX_ATTR, _attr_value_pb2.AttrValue(i=sort_index))
        if aggregate is not None:
            identity_op.op._set_attr(OpHint.FUNCTION_AGGREGATE_ATTR, _attr_value_pb2.AttrValue(s=_compat.as_bytes(aggregate)))
        return identity_op