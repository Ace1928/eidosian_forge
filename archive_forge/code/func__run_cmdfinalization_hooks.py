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
def _run_cmdfinalization_hooks(self, stop: bool, statement: Optional[Statement]) -> bool:
    """Run the command finalization hooks"""
    with self.sigint_protection:
        if not sys.platform.startswith('win') and self.stdin.isatty():
            import subprocess
            proc = subprocess.Popen(['stty', 'sane'])
            proc.communicate()
    data = plugin.CommandFinalizationData(stop, statement)
    for func in self._cmdfinalization_hooks:
        data = func(data)
    return data.stop