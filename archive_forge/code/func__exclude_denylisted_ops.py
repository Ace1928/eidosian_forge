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
def _exclude_denylisted_ops(self, node_names):
    """Exclude all nodes whose op types are in _GRAPH_STRUCT_OP_TYPE_DENYLIST.

    Args:
      node_names: An iterable of node or graph element names.

    Returns:
      A list of node names that are not denylisted.
    """
    return [node_name for node_name in node_names if self._debug_dump.node_op_type(debug_graphs.get_node_name(node_name)) not in self._GRAPH_STRUCT_OP_TYPE_DENYLIST]