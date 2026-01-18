import os
import stat
import sys
import textwrap
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union
from urllib.parse import urlparse
import click
import requests.utils
import wandb
from wandb.apis import InternalApi
from wandb.errors import term
from wandb.util import _is_databricks, isatty, prompt_choices
from .wburls import wburls
def _fixup_anon_mode(default: Optional[Mode]) -> Optional[Mode]:
    anon_mode = default or 'never'
    mapping: Dict[Mode, Mode] = {'true': 'allow', 'false': 'never'}
    return mapping.get(anon_mode, anon_mode)