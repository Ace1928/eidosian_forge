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
def _compare_run_metadata(self, src_run: Run, dst_run: Run) -> dict:
    fname = 'wandb-metadata.json'
    src_f = src_run.file(fname)
    if src_f.size == 0:
        return {}
    dst_f = dst_run.file(fname)
    try:
        contents = wandb.util.download_file_into_memory(dst_f.url, self.dst_api.api_key)
    except urllib3.exceptions.ReadTimeoutError:
        return {'Error checking': 'Timeout'}
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return {'Bad upload': f'File not found: {fname}'}
        return {'http problem': f'{fname}: ({e})'}
    dst_meta = wandb.wandb_sdk.lib.json_util.loads(contents)
    non_matching = {}
    if src_run.metadata:
        for k, src_v in src_run.metadata.items():
            if k not in dst_meta:
                non_matching[k] = {'src': src_v, 'dst': 'KEY NOT FOUND'}
                continue
            dst_v = dst_meta[k]
            if src_v != dst_v:
                non_matching[k] = {'src': src_v, 'dst': dst_v}
    return non_matching