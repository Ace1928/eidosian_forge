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
List dumped tensor data from a node.

    Args:
      node_name: Name of the node of which the attributes are to be listed.

    Returns:
      A RichTextLines object.
    