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
def _build_command_info(self) -> Tuple[Dict[str, List[str]], List[str], List[str], List[str]]:
    help_topics = sorted(self.get_help_topics(), key=self.default_sort_key)
    visible_commands = sorted(self.get_visible_commands(), key=self.default_sort_key)
    cmds_doc: List[str] = []
    cmds_undoc: List[str] = []
    cmds_cats: Dict[str, List[str]] = {}
    for command in visible_commands:
        func = self.cmd_func(command)
        has_help_func = False
        if command in help_topics:
            help_topics.remove(command)
            if not hasattr(func, constants.CMD_ATTR_ARGPARSER):
                has_help_func = True
        if hasattr(func, constants.CMD_ATTR_HELP_CATEGORY):
            category: str = getattr(func, constants.CMD_ATTR_HELP_CATEGORY)
            cmds_cats.setdefault(category, [])
            cmds_cats[category].append(command)
        elif func.__doc__ or has_help_func:
            cmds_doc.append(command)
        else:
            cmds_undoc.append(command)
    return (cmds_cats, cmds_doc, cmds_undoc, help_topics)