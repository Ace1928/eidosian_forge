import importlib
import logging
import sys
import textwrap
from functools import wraps
from typing import Any, Callable, Iterable, Optional, TypeVar, Union
from packaging.version import Version
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import DeveloperAPI
from ray.widgets import Template
def _get_ipython_shell_name() -> str:
    if 'IPython' in sys.modules:
        from IPython import get_ipython
        return get_ipython().__class__.__name__
    return ''