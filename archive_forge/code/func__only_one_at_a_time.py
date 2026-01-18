from __future__ import annotations
import asyncio
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections import deque
from enum import Enum
from functools import wraps
from typing import Any, Callable, Coroutine, Iterable, TypeVar, cast
from .application.current import get_app
from .application.run_in_terminal import run_in_terminal
from .auto_suggest import AutoSuggest, Suggestion
from .cache import FastDictCache
from .clipboard import ClipboardData
from .completion import (
from .document import Document
from .eventloop import aclosing
from .filters import FilterOrBool, to_filter
from .history import History, InMemoryHistory
from .search import SearchDirection, SearchState
from .selection import PasteMode, SelectionState, SelectionType
from .utils import Event, to_str
from .validation import ValidationError, Validator
def _only_one_at_a_time(coroutine: _T) -> _T:
    """
    Decorator that only starts the coroutine only if the previous call has
    finished. (Used to make sure that we have only one autocompleter, auto
    suggestor and validator running at a time.)

    When the coroutine raises `_Retry`, it is restarted.
    """
    running = False

    @wraps(coroutine)
    async def new_coroutine(*a: Any, **kw: Any) -> Any:
        nonlocal running
        if running:
            return
        running = True
        try:
            while True:
                try:
                    await coroutine(*a, **kw)
                except _Retry:
                    continue
                else:
                    return None
        finally:
            running = False
    return cast(_T, new_coroutine)