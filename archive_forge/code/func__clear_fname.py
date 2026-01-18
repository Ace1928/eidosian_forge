import itertools
import json
import logging
import numbers
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from unittest.mock import patch
import filelock
import polars as pl
import requests
import urllib3
import yaml
from wandb_gql import gql
import wandb
import wandb.apis.reports as wr
from wandb.apis.public import ArtifactCollection, Run
from wandb.apis.public.files import File
from wandb.apis.reports import Report
from wandb.util import coalesce, remove_keys_with_none_values
from . import validation
from .internals import internal
from .internals.protocols import PathStr, Policy
from .internals.util import Namespace, for_each
def _clear_fname(fname: str) -> None:
    old_fname = f'{internal.ROOT_DIR}/{fname}'
    new_fname = f'{internal.ROOT_DIR}/prev_{fname}'
    logger.debug(f'Moving old_fname={old_fname!r} to new_fname={new_fname!r}')
    try:
        shutil.copy2(old_fname, new_fname)
    except FileNotFoundError:
        pass
    with open(fname, 'w'):
        pass