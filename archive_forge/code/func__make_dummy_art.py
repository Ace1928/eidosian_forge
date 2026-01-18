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
def _make_dummy_art(name: str, _type: str, ver: int):
    art = Artifact(name, ART_DUMMY_PLACEHOLDER_TYPE)
    art._type = _type
    art._description = ART_SEQUENCE_DUMMY_PLACEHOLDER
    p = Path(ART_DUMMY_PLACEHOLDER_PATH)
    p.mkdir(parents=True, exist_ok=True)
    fname = p / str(ver)
    with open(fname, 'w'):
        pass
    art.add_file(fname)
    return art