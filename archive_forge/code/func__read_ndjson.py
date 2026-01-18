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
def _read_ndjson(fname: str) -> Optional[pl.DataFrame]:
    try:
        df = pl.read_ndjson(fname)
    except FileNotFoundError:
        return None
    except RuntimeError as e:
        if 'empty string is not a valid JSON value' in str(e):
            return None
        if 'error parsing ndjson' in str(e):
            return None
        raise e
    return df