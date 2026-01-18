import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
class RedirectionSavedState:
    """Created by each command to store information required to restore state after redirection"""

    def __init__(self, self_stdout: Union[StdSim, TextIO], sys_stdout: Union[StdSim, TextIO], pipe_proc_reader: Optional[ProcReader], saved_redirecting: bool) -> None:
        """
        RedirectionSavedState initializer
        :param self_stdout: saved value of Cmd.stdout
        :param sys_stdout: saved value of sys.stdout
        :param pipe_proc_reader: saved value of Cmd._cur_pipe_proc_reader
        :param saved_redirecting: saved value of Cmd._redirecting
        """
        self.redirecting = False
        self.saved_self_stdout = self_stdout
        self.saved_sys_stdout = sys_stdout
        self.saved_pipe_proc_reader = pipe_proc_reader
        self.saved_redirecting = saved_redirecting