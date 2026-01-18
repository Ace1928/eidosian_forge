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
Add our host and key to .netrc.