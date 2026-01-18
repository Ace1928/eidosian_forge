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
@staticmethod
def _determine_ap_completer_type(parser: argparse.ArgumentParser) -> Type[argparse_completer.ArgparseCompleter]:
    """
        Determine what type of ArgparseCompleter to use on a given parser. If the parser does not have one
        set, then use argparse_completer.DEFAULT_AP_COMPLETER.

        :param parser: the parser to examine
        :return: type of ArgparseCompleter
        """
    completer_type: Optional[Type[argparse_completer.ArgparseCompleter]] = parser.get_ap_completer_type()
    if completer_type is None:
        completer_type = argparse_completer.DEFAULT_AP_COMPLETER
    return completer_type