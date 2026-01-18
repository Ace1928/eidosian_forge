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
def _get_rows_from_parquet_history_paths(self) -> Iterable[Dict[str, Any]]:
    if not (paths := self._get_parquet_history_paths()):
        yield {}
        return
    dfs = [pl.read_parquet(p) for path in paths for p in Path(path).glob('*.parquet')]
    if '_step' in (df := _merge_dfs(dfs)):
        df = df.with_columns(pl.col('_step').cast(pl.Int64))
    yield from df.iter_rows(named=True)