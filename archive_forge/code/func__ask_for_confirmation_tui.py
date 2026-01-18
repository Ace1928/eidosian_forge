import os
from argparse import Namespace, _SubParsersAction
from functools import wraps
from tempfile import mkstemp
from typing import Any, Callable, Iterable, List, Optional, Union
from ..utils import CachedRepoInfo, CachedRevisionInfo, HFCacheInfo, scan_cache_dir
from . import BaseHuggingfaceCLICommand
from ._cli_utils import ANSI
@require_inquirer_py
def _ask_for_confirmation_tui(message: str, default: bool=True) -> bool:
    """Ask for confirmation using Inquirer."""
    return inquirer.confirm(message, default=default).execute()