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
def _use_artifact_sequence(self, sequence: ArtifactSequence, *, namespace: Optional[Namespace]=None):
    if namespace is None:
        namespace = Namespace(sequence.entity, sequence.project)
    settings_override = {'api_key': self.dst_api_key, 'base_url': self.dst_base_url, 'resume': 'true', 'resumed': True}
    logger.debug(f'Using artifact sequence with settings_override={settings_override!r}, namespace={namespace!r}')
    send_manager_config = internal.SendManagerConfig(use_artifacts=True)
    for art in sequence:
        if (used_by := art.used_by()) is None:
            continue
        for wandb_run in used_by:
            run = WandbRun(wandb_run, **self.run_api_kwargs)
            internal.send_run(run, overrides=namespace.send_manager_overrides, settings_override=settings_override, config=send_manager_config)