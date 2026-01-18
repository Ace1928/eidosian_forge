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
def _validate_runs(self, runs: Iterable[WandbRun], *, skip_previously_validated: bool=True, remapping: Optional[Dict[Namespace, Namespace]]=None):
    base_runs = [r.run for r in runs]
    if skip_previously_validated:
        base_runs = list(self._filter_previously_checked_runs(base_runs, remapping=remapping))

    def _validate_run(run):
        logger.debug(f'Validating run={run!r}')
        self._validate_run(run, remapping=remapping)
        logger.debug(f'Finished validating run={run!r}')
    for_each(_validate_run, base_runs)