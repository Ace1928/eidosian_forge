import itertools
import platform
import sys
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union
import click
import wandb
from . import ipython, sparkline
@staticmethod
def _display_fn_mapping(level: Optional[Union[str, int]]) -> Callable[[str], None]:
    level = _Printer._sanitize_level(level)
    if level >= CRITICAL:
        return ipython.display_html
    elif ERROR <= level < CRITICAL:
        return ipython.display_html
    elif WARNING <= level < ERROR:
        return ipython.display_html
    elif INFO <= level < WARNING:
        return ipython.display_html
    elif DEBUG <= level < INFO:
        return ipython.display_html
    else:
        return ipython.display_html