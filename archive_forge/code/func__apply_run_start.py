import collections.abc
import configparser
import enum
import getpass
import json
import logging
import multiprocessing
import os
import platform
import re
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
from functools import reduce
from typing import (
from urllib.parse import quote, unquote, urlencode, urlparse, urlsplit
from google.protobuf.wrappers_pb2 import BoolValue, DoubleValue, Int32Value, StringValue
import wandb
import wandb.env
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import UsageError
from wandb.proto import wandb_settings_pb2
from wandb.sdk.internal.system.env_probe_helpers import is_aws_lambda
from wandb.sdk.lib import filesystem
from wandb.sdk.lib._settings_toposort_generated import SETTINGS_TOPOLOGICALLY_SORTED
from wandb.sdk.wandb_setup import _EarlyLogger
from .lib import apikey
from .lib.gitlib import GitRepo
from .lib.ipython import _get_python_type
from .lib.runid import generate_id
def _apply_run_start(self, run_start_settings: Dict[str, Any]) -> None:
    param_map = {'run_id': 'run_id', 'entity': 'entity', 'project': 'project', 'run_group': 'run_group', 'job_type': 'run_job_type', 'display_name': 'run_name', 'notes': 'run_notes', 'tags': 'run_tags', 'sweep_id': 'sweep_id', 'host': 'host', 'resumed': 'resumed', 'git.remote_url': 'git_remote_url', 'git.commit': 'git_commit'}
    run_settings = {name: reduce(lambda d, k: d.get(k, {}), attr.split('.'), run_start_settings) for attr, name in param_map.items()}
    run_settings = {key: value for key, value in run_settings.items() if value}
    if run_settings:
        self.update(run_settings, source=Source.RUN)