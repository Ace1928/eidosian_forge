import optparse
import os
import sys
from itertools import chain
from typing import Any, Iterable, List, Optional
from pip._internal.cli.main_parser import create_main_parser
from pip._internal.commands import commands_dict, create_command
from pip._internal.metadata import get_default_environment
If ``completion_type`` is ``file`` or ``path``, list all regular files
    and directories starting with ``current``; otherwise only list directories
    starting with ``current``.

    :param current: The word to be completed
    :param completion_type: path completion type(``file``, ``path`` or ``dir``)
    :return: A generator of regular files and/or directories
    