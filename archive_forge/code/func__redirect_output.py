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
def _redirect_output(self, statement: Statement) -> utils.RedirectionSavedState:
    """Set up a command's output redirection for >, >>, and |.

        :param statement: a parsed statement from the user
        :return: A bool telling if an error occurred and a utils.RedirectionSavedState object
        :raises: RedirectionError if an error occurs trying to pipe or redirect
        """
    import io
    import subprocess
    redir_saved_state = utils.RedirectionSavedState(cast(TextIO, self.stdout), sys.stdout, self._cur_pipe_proc_reader, self._redirecting)
    cmd_pipe_proc_reader: Optional[utils.ProcReader] = None
    if not self.allow_redirection:
        pass
    elif statement.pipe_to:
        read_fd, write_fd = os.pipe()
        subproc_stdin = io.open(read_fd, 'r')
        new_stdout: TextIO = cast(TextIO, io.open(write_fd, 'w'))
        kwargs: Dict[str, Any] = dict()
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs['start_new_session'] = True
            shell = os.environ.get('SHELL')
            if shell:
                kwargs['executable'] = shell
        proc = subprocess.Popen(statement.pipe_to, stdin=subproc_stdin, stdout=subprocess.PIPE if isinstance(self.stdout, utils.StdSim) else self.stdout, stderr=subprocess.PIPE if isinstance(sys.stderr, utils.StdSim) else sys.stderr, shell=True, **kwargs)
        try:
            proc.wait(0.2)
        except subprocess.TimeoutExpired:
            pass
        if proc.returncode is not None:
            subproc_stdin.close()
            new_stdout.close()
            raise RedirectionError(f'Pipe process exited with code {proc.returncode} before command could run')
        else:
            redir_saved_state.redirecting = True
            cmd_pipe_proc_reader = utils.ProcReader(proc, cast(TextIO, self.stdout), sys.stderr)
            sys.stdout = self.stdout = new_stdout
    elif statement.output:
        import tempfile
        if not statement.output_to and (not self._can_clip):
            raise RedirectionError("Cannot redirect to paste buffer; missing 'pyperclip' and/or pyperclip dependencies")
        elif statement.output_to:
            mode = 'a' if statement.output == constants.REDIRECTION_APPEND else 'w'
            try:
                new_stdout = cast(TextIO, open(utils.strip_quotes(statement.output_to), mode=mode, buffering=1))
            except OSError as ex:
                raise RedirectionError(f'Failed to redirect because: {ex}')
            redir_saved_state.redirecting = True
            sys.stdout = self.stdout = new_stdout
        else:
            new_stdout = cast(TextIO, tempfile.TemporaryFile(mode='w+'))
            redir_saved_state.redirecting = True
            sys.stdout = self.stdout = new_stdout
            if statement.output == constants.REDIRECTION_APPEND:
                self.stdout.write(get_paste_buffer())
                self.stdout.flush()
    self._cur_pipe_proc_reader = cmd_pipe_proc_reader
    self._redirecting = redir_saved_state.redirecting
    return redir_saved_state