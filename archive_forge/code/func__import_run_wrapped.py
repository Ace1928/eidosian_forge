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
def _import_run_wrapped(run):
    namespace = Namespace(run.entity(), run.project())
    if remapping is not None and namespace in remapping:
        namespace = remapping[namespace]
    config = internal.SendManagerConfig(metadata=metadata, files=files, media=media, code=code, history=history, summary=summary, terminal_output=terminal_output)
    logger.debug(f'Importing run={run!r}, namespace={namespace!r}, config={config!r}')
    self._import_run(run, namespace=namespace, config=config)
    logger.debug(f'Finished importing run={run!r}, namespace={namespace!r}, config={config!r}')