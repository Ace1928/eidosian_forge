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
def _reconstruct_print_source_command(self, parsed, line_begin, max_elements_per_line_increase=0):
    return 'ps %s %s -b %d -m %d' % (parsed.source_file_path, '-t' if parsed.tensors else '', line_begin, parsed.max_elements_per_line + max_elements_per_line_increase)