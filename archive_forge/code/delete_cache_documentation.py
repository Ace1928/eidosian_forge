import os
from argparse import Namespace, _SubParsersAction
from functools import wraps
from tempfile import mkstemp
from typing import Any, Callable, Iterable, List, Optional, Union
from ..utils import CachedRepoInfo, CachedRevisionInfo, HFCacheInfo, scan_cache_dir
from . import BaseHuggingfaceCLICommand
from ._cli_utils import ANSI
Run `delete-cache` command with or without TUI.