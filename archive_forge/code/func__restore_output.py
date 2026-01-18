import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def _restore_output(self, statement: Statement, saved_redir_state: utils.RedirectionSavedState) -> None:
    """Handles restoring state after output redirection

        :param statement: Statement object which contains the parsed input from the user
        :param saved_redir_state: contains information needed to restore state data
        """
    if saved_redir_state.redirecting:
        if statement.output and (not statement.output_to):
            self.stdout.seek(0)
            write_to_paste_buffer(self.stdout.read())
        try:
            self.stdout.close()
        except BrokenPipeError:
            pass
        self.stdout = cast(TextIO, saved_redir_state.saved_self_stdout)
        sys.stdout = cast(TextIO, saved_redir_state.saved_sys_stdout)
        if self._cur_pipe_proc_reader is not None:
            self._cur_pipe_proc_reader.wait()
    self._cur_pipe_proc_reader = saved_redir_state.saved_pipe_proc_reader
    self._redirecting = saved_redir_state.saved_redirecting