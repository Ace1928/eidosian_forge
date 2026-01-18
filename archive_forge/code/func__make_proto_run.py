import _thread as thread
import atexit
import functools
import glob
import json
import logging
import numbers
import os
import re
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from types import TracebackType
from typing import (
import requests
import wandb
import wandb.env
from wandb import errors, trigger
from wandb._globals import _datatypes_set_callback
from wandb.apis import internal, public
from wandb.apis.internal import Api
from wandb.apis.public import Api as PublicApi
from wandb.proto.wandb_internal_pb2 import (
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.internal import job_builder
from wandb.sdk.lib.import_hooks import (
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath
from wandb.util import (
from wandb.viz import CustomChart, Visualize, custom_chart
from . import wandb_config, wandb_metric, wandb_summary
from .data_types._dtypes import TypeRegistry
from .interface.interface import GlobStr, InterfaceBase
from .interface.summary_record import SummaryRecord
from .lib import (
from .lib.exit_hooks import ExitHooks
from .lib.gitlib import GitRepo
from .lib.mailbox import MailboxError, MailboxHandle, MailboxProbe, MailboxProgress
from .lib.printer import get_printer
from .lib.proto_util import message_to_dict
from .lib.reporting import Reporter
from .lib.wburls import wburls
from .wandb_settings import Settings
from .wandb_setup import _WandbSetup
def _make_proto_run(self, run: RunRecord) -> None:
    """Populate protocol buffer RunData for interface/interface."""
    if self._entity is not None:
        run.entity = self._entity
    if self._project is not None:
        run.project = self._project
    if self._group is not None:
        run.run_group = self._group
    if self._job_type is not None:
        run.job_type = self._job_type
    if self._run_id is not None:
        run.run_id = self._run_id
    if self._name is not None:
        run.display_name = self._name
    if self._notes is not None:
        run.notes = self._notes
    if self._tags is not None:
        for tag in self._tags:
            run.tags.append(tag)
    if self._start_time is not None:
        run.start_time.FromMicroseconds(int(self._start_time * 1000000.0))
    if self._remote_url is not None:
        run.git.remote_url = self._remote_url
    if self._commit is not None:
        run.git.commit = self._commit
    if self._sweep_id is not None:
        run.sweep_id = self._sweep_id