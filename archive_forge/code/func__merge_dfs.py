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
def _merge_dfs(dfs: List[pl.DataFrame]) -> pl.DataFrame:
    if len(dfs) == 0:
        return pl.DataFrame()
    if len(dfs) == 1:
        return dfs[0]
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.join(df, how='outer', on=['_step'])
        col_pairs = [(c, f'{c}_right') for c in merged_df.columns if f'{c}_right' in merged_df.columns]
        for col, right in col_pairs:
            new_col = merged_df[col].fill_null(merged_df[right])
            merged_df = merged_df.with_columns(new_col).drop(right)
    return merged_df