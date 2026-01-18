import itertools
import platform
import sys
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union
import click
import wandb
from . import ipython, sparkline
def _display(self, text: Union[str, List[str], Tuple[str]], *, level: Optional[Union[str, int]]=None, default_text: Optional[Union[str, List[str], Tuple[str]]]=None) -> None:
    text = '<br/>'.join(text) if isinstance(text, (list, tuple)) else text
    if default_text is not None:
        default_text = '<br/>'.join(default_text) if isinstance(default_text, (list, tuple)) else default_text
        text = text or default_text
    self._display_fn_mapping(level)(text)