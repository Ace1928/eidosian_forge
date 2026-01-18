import logging
import os
import subprocess
from optparse import Values
from typing import Any, List, Optional
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.configuration import (
from pip._internal.exceptions import PipError
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import get_prog, write_output
def _get_n_args(self, args: List[str], example: str, n: int) -> Any:
    """Helper to make sure the command got the right number of arguments"""
    if len(args) != n:
        msg = f'Got unexpected number of arguments, expected {n}. (example: "{get_prog()} config {example}")'
        raise PipError(msg)
    if n == 1:
        return args[0]
    else:
        return args