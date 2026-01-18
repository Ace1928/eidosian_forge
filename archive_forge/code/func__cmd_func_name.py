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
def _cmd_func_name(self, command: str) -> str:
    """Get the method name associated with a given command.

        :param command: command to look up method name which implements it
        :return: method name which implements the given command
        """
    target = constants.COMMAND_FUNC_PREFIX + command
    return target if callable(getattr(self, target, None)) else ''