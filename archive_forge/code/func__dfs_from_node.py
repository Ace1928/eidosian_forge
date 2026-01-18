import argparse
import copy
import re
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import evaluator
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import source_utils
def _dfs_from_node(self, lines, attr_segs, node_name, tracker, max_depth, depth, unfinished, include_control=False, show_op_type=False, command_template=None):
    """Perform depth-first search (DFS) traversal of a node's input tree.

    It recursively tracks the inputs (or output recipients) of the node called
    node_name, and append these inputs (or output recipients) to a list of text
    lines (lines) with proper indentation that reflects the recursion depth,
    together with some formatting attributes (to attr_segs). The formatting
    attributes can include command shortcuts, for example.

    Args:
      lines: Text lines to append to, as a list of str.
      attr_segs: (dict) Attribute segments dictionary to append to.
      node_name: Name of the node, as a str. This arg is updated during the
        recursion.
      tracker: A callable that takes one str as the node name input and
        returns a list of str as the inputs/outputs.
        This makes it this function general enough to be used with both
        node-input and node-output tracking.
      max_depth: Maximum recursion depth, as an int.
      depth: Current recursion depth. This arg is updated during the
        recursion.
      unfinished: A stack of unfinished recursion depths, as a list of int.
      include_control: Whether control dependencies are to be included as
        inputs (and marked as such).
      show_op_type: Whether op type of the input nodes are to be displayed
        alongside the nodes' names.
      command_template: (str) Template for command shortcut of the node names.
    """
    all_inputs = self._exclude_denylisted_ops(copy.copy(tracker(node_name, is_control=False)))
    is_ctrl = [False] * len(all_inputs)
    if include_control:
        ctrl_inputs = self._exclude_denylisted_ops(sorted(tracker(node_name, is_control=True)))
        all_inputs.extend(ctrl_inputs)
        is_ctrl.extend([True] * len(ctrl_inputs))
    if not all_inputs:
        if depth == 1:
            lines.append('  [None]')
        return
    unfinished.append(depth)
    hang = ''
    for k in range(depth):
        if k < depth - 1:
            if k + 1 in unfinished:
                hang += HANG_UNFINISHED
            else:
                hang += HANG_FINISHED
        else:
            hang += HANG_SUFFIX
    if all_inputs and depth > max_depth:
        lines.append(hang + ELLIPSIS)
        unfinished.pop()
        return
    hang += DEPTH_TEMPLATE % depth
    for i, inp in enumerate(all_inputs):
        op_type = self._debug_dump.node_op_type(debug_graphs.get_node_name(inp))
        if op_type in self._GRAPH_STRUCT_OP_TYPE_DENYLIST:
            continue
        if is_ctrl[i]:
            ctrl_str = CTRL_LABEL
        else:
            ctrl_str = ''
        op_type_str = ''
        if show_op_type:
            op_type_str = OP_TYPE_TEMPLATE % op_type
        if i == len(all_inputs) - 1:
            unfinished.pop()
        line = hang + ctrl_str + op_type_str + inp
        lines.append(line)
        if command_template:
            attr_segs[len(lines) - 1] = [(len(line) - len(inp), len(line), debugger_cli_common.MenuItem(None, command_template % inp))]
        inp_node_name, _ = debug_graphs.parse_node_or_tensor_name(inp)
        self._dfs_from_node(lines, attr_segs, inp_node_name, tracker, max_depth, depth + 1, unfinished, include_control=include_control, show_op_type=show_op_type, command_template=command_template)