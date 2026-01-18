import os
import six
import sys
from pyparsing import (alphanums, Empty, Group, locatedExpr,
from . import console
from . import log
from . import prefs
from .node import ConfigNode, ExecutionError
import signal
def _cli_loop(self):
    """
        Starts the configuration shell interactive loop, that:
            - Goes to the last current path
            - Displays the prompt
            - Waits for user input
            - Runs user command
        """
    while not self._exit:
        try:
            readline.parse_and_bind('tab: complete')
            readline.set_completer(self._complete)
            cmdline = six.moves.input(self._get_prompt()).strip()
        except EOFError:
            self.con.raw_write('exit\n')
            cmdline = 'exit'
        self.run_cmdline(cmdline)
        if self._save_history:
            try:
                readline.write_history_file(self._cmd_history)
            except IOError:
                self.log.warning('Cannot write to command history file %s.' % self._cmd_history)
                self.log.warning('Saving command history has been disabled!')
                self._save_history = False