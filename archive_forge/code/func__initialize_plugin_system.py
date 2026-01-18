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
def _initialize_plugin_system(self) -> None:
    """Initialize the plugin system"""
    self._preloop_hooks: List[Callable[[], None]] = []
    self._postloop_hooks: List[Callable[[], None]] = []
    self._postparsing_hooks: List[Callable[[plugin.PostparsingData], plugin.PostparsingData]] = []
    self._precmd_hooks: List[Callable[[plugin.PrecommandData], plugin.PrecommandData]] = []
    self._postcmd_hooks: List[Callable[[plugin.PostcommandData], plugin.PostcommandData]] = []
    self._cmdfinalization_hooks: List[Callable[[plugin.CommandFinalizationData], plugin.CommandFinalizationData]] = []