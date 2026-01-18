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
@staticmethod
def _footer_local_warn(server_info_response: Optional[ServerInfoResponse]=None, quiet: Optional[bool]=None, *, settings: 'Settings', printer: Union['PrinterTerm', 'PrinterJupyter']) -> None:
    if (quiet or settings.quiet) or settings.silent:
        return
    if settings._offline:
        return
    if not server_info_response or not server_info_response.local_info:
        return
    if settings.is_local:
        local_info = server_info_response.local_info
        latest_version, out_of_date = (local_info.version, local_info.out_of_date)
        if out_of_date:
            printer.display(f'Upgrade to the {latest_version} version of W&B Server to get the latest features. Learn more: {printer.link(wburls.get('upgrade_server'))}', level='warn')