import readline
from tensorflow.python.debug.cli import base_ui
from tensorflow.python.debug.cli import debugger_cli_common
def _get_user_command(self):
    print('')
    return self._input(self.CLI_PROMPT).strip()