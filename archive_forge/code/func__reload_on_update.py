import os
import sys
import functools
import importlib.abc
import os
import pkgutil
import sys
import traceback
import types
import subprocess
import weakref
from tornado import ioloop
from tornado.log import gen_log
from tornado import process
from typing import Callable, Dict, Optional, List, Union
def _reload_on_update(modify_times: Dict[str, float]) -> None:
    if _reload_attempted:
        return
    if process.task_id() is not None:
        return
    for module in list(sys.modules.values()):
        if not isinstance(module, types.ModuleType):
            continue
        path = getattr(module, '__file__', None)
        if not path:
            continue
        if path.endswith('.pyc') or path.endswith('.pyo'):
            path = path[:-1]
        _check_file(modify_times, path)
    for path in _watched_files:
        _check_file(modify_times, path)