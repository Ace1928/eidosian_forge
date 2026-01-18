import itertools
import platform
import sys
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union
import click
import wandb
from . import ipython, sparkline
@staticmethod
def _sanitize_level(name_or_level: Optional[Union[str, int]]) -> int:
    if isinstance(name_or_level, str):
        try:
            return _name_to_level[name_or_level.upper()]
        except KeyError:
            raise ValueError(f'Unknown level name: {name_or_level}, supported levels: {_name_to_level.keys()}')
    if isinstance(name_or_level, int):
        return name_or_level
    if name_or_level is None:
        return INFO
    raise ValueError(f'Unknown status level {name_or_level}')