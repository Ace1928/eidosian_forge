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
def _prep_profile_cli_for_run_end(self, py_graph, run_metadata):
    self._init_command = 'lp'
    self._run_cli = profile_analyzer_cli.create_profiler_ui(py_graph, run_metadata, ui_type=self._ui_type, config=self._run_cli.config)
    self._title = 'run-end (profiler mode): ' + self._run_description