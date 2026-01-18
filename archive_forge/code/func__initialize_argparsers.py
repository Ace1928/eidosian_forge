import argparse
import os
import sys
import tempfile
from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import profile_analyzer_cli
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.lib.io import file_io
def _initialize_argparsers(self):
    self._argparsers = {}
    ap = argparse.ArgumentParser(description='Run through, with or without debug tensor watching.', usage=argparse.SUPPRESS)
    ap.add_argument('-t', '--times', dest='times', type=int, default=1, help='How many Session.run() calls to proceed with.')
    ap.add_argument('-n', '--no_debug', dest='no_debug', action='store_true', help='Run through without debug tensor watching.')
    ap.add_argument('-f', '--till_filter_pass', dest='till_filter_pass', type=str, default='', help='Run until a tensor in the graph passes the specified filter.')
    ap.add_argument('-fenn', '--filter_exclude_node_names', dest='filter_exclude_node_names', type=str, default='', help='When applying the tensor filter, exclude node with names matching the regular expression. Applicable only if --tensor_filter or -f is used.')
    ap.add_argument('--node_name_filter', dest='node_name_filter', type=str, default='', help='Regular-expression filter for node names to be watched in the run, e.g., loss, reshape.*')
    ap.add_argument('--op_type_filter', dest='op_type_filter', type=str, default='', help='Regular-expression filter for op type to be watched in the run, e.g., (MatMul|Add), Variable.*')
    ap.add_argument('--tensor_dtype_filter', dest='tensor_dtype_filter', type=str, default='', help='Regular-expression filter for tensor dtype to be watched in the run, e.g., (float32|float64), int.*')
    ap.add_argument('-p', '--profile', dest='profile', action='store_true', help='Run and profile TensorFlow graph execution.')
    self._argparsers['run'] = ap
    ap = argparse.ArgumentParser(description='Display information about this Session.run() call.', usage=argparse.SUPPRESS)
    self._argparsers['run_info'] = ap
    self._argparsers['print_feed'] = command_parser.get_print_tensor_argparser('Print the value of a feed in feed_dict.')