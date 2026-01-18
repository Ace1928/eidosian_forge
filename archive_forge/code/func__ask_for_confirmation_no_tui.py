import os
from argparse import Namespace, _SubParsersAction
from functools import wraps
from tempfile import mkstemp
from typing import Any, Callable, Iterable, List, Optional, Union
from ..utils import CachedRepoInfo, CachedRevisionInfo, HFCacheInfo, scan_cache_dir
from . import BaseHuggingfaceCLICommand
from ._cli_utils import ANSI
def _ask_for_confirmation_no_tui(message: str, default: bool=True) -> bool:
    """Ask for confirmation using pure-python."""
    YES = ('y', 'yes', '1')
    NO = ('n', 'no', '0')
    DEFAULT = ''
    ALL = YES + NO + (DEFAULT,)
    full_message = message + (' (Y/n) ' if default else ' (y/N) ')
    while True:
        answer = input(full_message).lower()
        if answer == DEFAULT:
            return default
        if answer in YES:
            return True
        if answer in NO:
            return False
        print(f'Invalid input. Must be one of {ALL}')