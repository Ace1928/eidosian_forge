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
class _SavedReadlineSettings:
    """readline settings that are backed up when switching between readline environments"""

    def __init__(self) -> None:
        self.completer = None
        self.delims = ''
        self.basic_quotes: Optional[bytes] = None