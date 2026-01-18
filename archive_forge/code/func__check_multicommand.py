import enum
import errno
import inspect
import os
import sys
import typing as t
from collections import abc
from contextlib import contextmanager
from contextlib import ExitStack
from functools import update_wrapper
from gettext import gettext as _
from gettext import ngettext
from itertools import repeat
from types import TracebackType
from . import types
from .exceptions import Abort
from .exceptions import BadParameter
from .exceptions import ClickException
from .exceptions import Exit
from .exceptions import MissingParameter
from .exceptions import UsageError
from .formatting import HelpFormatter
from .formatting import join_options
from .globals import pop_context
from .globals import push_context
from .parser import _flag_needs_value
from .parser import OptionParser
from .parser import split_opt
from .termui import confirm
from .termui import prompt
from .termui import style
from .utils import _detect_program_name
from .utils import _expand_args
from .utils import echo
from .utils import make_default_short_help
from .utils import make_str
from .utils import PacifyFlushWrapper
def _check_multicommand(base_command: 'MultiCommand', cmd_name: str, cmd: 'Command', register: bool=False) -> None:
    if not base_command.chain or not isinstance(cmd, MultiCommand):
        return
    if register:
        hint = 'It is not possible to add multi commands as children to another multi command that is in chain mode.'
    else:
        hint = 'Found a multi command as subcommand to a multi command that is in chain mode. This is not supported.'
    raise RuntimeError(f'{hint}. Command {base_command.name!r} is set to chain and {cmd_name!r} was added as a subcommand but it in itself is a multi command. ({cmd_name!r} is a {type(cmd).__name__} within a chained {type(base_command).__name__} named {base_command.name!r}).')