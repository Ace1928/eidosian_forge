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
def _exit_if_requested_by_user(self):
    if self._run_start_response == debugger_cli_common.EXPLICIT_USER_EXIT:
        print('Note: user exited from debugger CLI: Calling sys.exit(1).', file=sys.stderr)
        sys.exit(1)