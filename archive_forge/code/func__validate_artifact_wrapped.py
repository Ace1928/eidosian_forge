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
def _validate_artifact_wrapped(args):
    art, entity, project = args
    if remapping is not None and (namespace := Namespace(entity, project)) in remapping:
        remapped_ns = remapping[namespace]
        entity = remapped_ns.entity
        project = remapped_ns.project
    logger.debug(f'Validating art={art!r}, entity={entity!r}, project={project!r}')
    result = self._validate_artifact(art, entity, project, download_files_and_compare=download_files_and_compare, check_entries_are_downloadable=check_entries_are_downloadable)
    logger.debug(f'Finished validating art={art!r}, entity={entity!r}, project={project!r}')
    return result