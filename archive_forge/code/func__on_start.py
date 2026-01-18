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
def _on_start(self) -> None:
    self._set_globals()
    self._header(self._check_version, settings=self._settings, printer=self._printer)
    if self._settings.save_code and self._settings.code_dir is not None:
        self.log_code(self._settings.code_dir)
    if self._settings._save_requirements:
        if self._backend and self._backend.interface:
            from wandb.util import working_set
            logger.debug('Saving list of pip packages installed into the current environment')
            self._backend.interface.publish_python_packages(working_set())
    if self._backend and self._backend.interface and (not self._settings._offline):
        self._run_status_checker = RunStatusChecker(interface=self._backend.interface)
        self._run_status_checker.start()
    self._console_start()
    self._on_ready()