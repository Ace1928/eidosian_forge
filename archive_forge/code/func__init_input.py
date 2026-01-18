import readline
from tensorflow.python.debug.cli import base_ui
from tensorflow.python.debug.cli import debugger_cli_common
def _init_input(self):
    readline.parse_and_bind('set editing-mode emacs')
    readline.set_completer_delims('\n')
    readline.set_completer(self._readline_complete)
    readline.parse_and_bind('tab: complete')
    self._input = input