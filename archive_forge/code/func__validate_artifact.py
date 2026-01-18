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
def _validate_artifact(self, src_art: Artifact, dst_entity: str, dst_project: str, download_files_and_compare: bool=False, check_entries_are_downloadable: bool=True):
    problems = []
    ignore_patterns = ['^job-(.*?)\\.py(:v\\d+)?$']
    for pattern in ignore_patterns:
        if re.search(pattern, src_art.name):
            return (src_art, dst_entity, dst_project, problems)
    try:
        dst_art = self._get_dst_art(src_art, dst_entity, dst_project)
    except Exception:
        problems.append('destination artifact not found')
        return (src_art, dst_entity, dst_project, problems)
    try:
        logger.debug('Comparing artifact manifests')
    except Exception as e:
        problems.append(f'Problem getting problems! problem with src_art.entity={src_art.entity!r}, src_art.project={src_art.project!r}, src_art.name={src_art.name!r} e={e!r}')
    else:
        problems += validation._compare_artifact_manifests(src_art, dst_art)
    if check_entries_are_downloadable:
        validation._check_entries_are_downloadable(dst_art)
    if download_files_and_compare:
        logger.debug(f'Downloading src_art={src_art!r}')
        try:
            src_dir = _download_art(src_art, root=f'{SRC_ART_PATH}/{src_art.name}')
        except requests.HTTPError as e:
            problems.append(f'Invalid download link for src src_art.entity={src_art.entity!r}, src_art.project={src_art.project!r}, src_art.name={src_art.name!r}, {e}')
        logger.debug(f'Downloading dst_art={dst_art!r}')
        try:
            dst_dir = _download_art(dst_art, root=f'{DST_ART_PATH}/{dst_art.name}')
        except requests.HTTPError as e:
            problems.append(f'Invalid download link for dst dst_art.entity={dst_art.entity!r}, dst_art.project={dst_art.project!r}, dst_art.name={dst_art.name!r}, {e}')
        else:
            logger.debug(f'Comparing artifact dirs src_dir={src_dir!r}, dst_dir={dst_dir!r}')
            if (problem := validation._compare_artifact_dirs(src_dir, dst_dir)):
                problems.append(problem)
    return (src_art, dst_entity, dst_project, problems)