import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
class CommandHistory:
    """Keeps command history and supports lookup."""
    _HISTORY_FILE_NAME = '.tfdbg_history'

    def __init__(self, limit=100, history_file_path=None):
        """CommandHistory constructor.

    Args:
      limit: Maximum number of the most recent commands that this instance
        keeps track of, as an int.
      history_file_path: (str) Manually specified path to history file. Used in
        testing.
    """
        self._commands = []
        self._limit = limit
        self._history_file_path = history_file_path or self._get_default_history_file_path()
        self._load_history_from_file()

    def _load_history_from_file(self):
        if os.path.isfile(self._history_file_path):
            try:
                with open(self._history_file_path, 'rt') as history_file:
                    commands = history_file.readlines()
                self._commands = [command.strip() for command in commands if command.strip()]
                if len(self._commands) > self._limit:
                    self._commands = self._commands[-self._limit:]
                    with open(self._history_file_path, 'wt') as history_file:
                        for command in self._commands:
                            history_file.write(command + '\n')
            except IOError:
                print('WARNING: writing history file failed.')

    def _add_command_to_history_file(self, command):
        try:
            with open(self._history_file_path, 'at') as history_file:
                history_file.write(command + '\n')
        except IOError:
            pass

    @classmethod
    def _get_default_history_file_path(cls):
        return os.path.join(os.path.expanduser('~'), cls._HISTORY_FILE_NAME)

    def add_command(self, command):
        """Add a command to the command history.

    Args:
      command: The history command, as a str.

    Raises:
      TypeError: if command is not a str.
    """
        if self._commands and command == self._commands[-1]:
            return
        if not isinstance(command, str):
            raise TypeError('Attempt to enter non-str entry to command history')
        self._commands.append(command)
        if len(self._commands) > self._limit:
            self._commands = self._commands[-self._limit:]
        self._add_command_to_history_file(command)

    def most_recent_n(self, n):
        """Look up the n most recent commands.

    Args:
      n: Number of most recent commands to look up.

    Returns:
      A list of n most recent commands, or all available most recent commands,
      if n exceeds size of the command history, in chronological order.
    """
        return self._commands[-n:]

    def lookup_prefix(self, prefix, n):
        """Look up the n most recent commands that starts with prefix.

    Args:
      prefix: The prefix to lookup.
      n: Number of most recent commands to look up.

    Returns:
      A list of n most recent commands that have the specified prefix, or all
      available most recent commands that have the prefix, if n exceeds the
      number of history commands with the prefix.
    """
        commands = [cmd for cmd in self._commands if cmd.startswith(prefix)]
        return commands[-n:]