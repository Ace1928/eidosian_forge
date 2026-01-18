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
def _compare_run_summary(self, src_run: Run, dst_run: Run) -> dict:
    non_matching = {}
    for k, src_v in src_run.summary.items():
        if isinstance(src_v, str) and src_v.startswith('wandb-client-artifact://'):
            continue
        if k in ('_wandb', '_runtime'):
            continue
        src_v = _recursive_cast_to_dict(src_v)
        dst_v = dst_run.summary.get(k)
        dst_v = _recursive_cast_to_dict(dst_v)
        if isinstance(src_v, dict) and isinstance(dst_v, dict):
            for kk, sv in src_v.items():
                if isinstance(sv, str) and sv.startswith('wandb-client-artifact://'):
                    continue
                dv = dst_v.get(kk)
                if not _almost_equal(sv, dv):
                    non_matching[f'{k}-{kk}'] = {'src': sv, 'dst': dv}
        elif not _almost_equal(src_v, dst_v):
            non_matching[k] = {'src': src_v, 'dst': dst_v}
    return non_matching