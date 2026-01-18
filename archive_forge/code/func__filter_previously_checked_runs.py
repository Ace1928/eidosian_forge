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
def _filter_previously_checked_runs(self, runs: Iterable[Run], *, remapping: Optional[Dict[Namespace, Namespace]]=None) -> Iterable[Run]:
    if (df := _read_ndjson(RUN_SUCCESSES_FNAME)) is None:
        logger.debug(f'RUN_SUCCESSES_FNAME={RUN_SUCCESSES_FNAME!r} is empty, yielding all runs')
        yield from runs
        return
    data = []
    for r in runs:
        namespace = Namespace(r.entity, r.project)
        if remapping is not None and namespace in remapping:
            namespace = remapping[namespace]
        data.append({'src_entity': r.entity, 'src_project': r.project, 'dst_entity': namespace.entity, 'dst_project': namespace.project, 'run_id': r.id, 'data': r})
    df2 = pl.DataFrame(data)
    logger.debug(f'Starting with len(runs)={len(runs)!r} in namespaces')
    results = df2.join(df, how='anti', on=['src_entity', 'src_project', 'dst_entity', 'dst_project', 'run_id'])
    logger.debug(f'After filtering out already successful runs, len(results)={len(results)!r}')
    if not results.is_empty():
        results = results.filter(~results['run_id'].is_null())
        results = results.unique(['src_entity', 'src_project', 'dst_entity', 'dst_project', 'run_id'])
    for r in results.iter_rows(named=True):
        yield r['data']