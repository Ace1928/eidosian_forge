import asyncio
import configparser
import datetime
import getpass
import json
import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from functools import wraps
from typing import Any, Dict, Optional
import click
import yaml
from click.exceptions import ClickException
from dockerpycreds.utils import find_executable
import wandb
import wandb.env
import wandb.sdk.verify.verify as wandb_verify
from wandb import Config, Error, env, util, wandb_agent, wandb_sdk
from wandb.apis import InternalApi, PublicApi
from wandb.apis.public import RunQueue
from wandb.integration.magic import magic_install
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.launch import utils as launch_utils
from wandb.sdk.launch._launch_add import _launch_add
from wandb.sdk.launch.errors import ExecutionError, LaunchError
from wandb.sdk.launch.sweeps import utils as sweep_utils
from wandb.sdk.launch.sweeps.scheduler import Scheduler
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.wburls import wburls
from wandb.sync import SyncManager, get_run_from_path, get_runs
import __main__
def _sync_path(_path, _sync_tensorboard):
    if run_id and len(_path) > 1:
        wandb.termerror('id can only be set for a single run.')
        sys.exit(1)
    sm = SyncManager(project=project, entity=entity, run_id=run_id, job_type=job_type, mark_synced=mark_synced, app_url=api.app_url, view=view, verbose=verbose, sync_tensorboard=_sync_tensorboard, log_path=_wandb_log_path, append=append, skip_console=skip_console)
    for p in _path:
        sm.add(p)
    sm.start()
    while not sm.is_done():
        _ = sm.poll()