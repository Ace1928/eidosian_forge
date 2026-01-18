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
def _footer_single_run_file_pusher_status_info(poll_exit_response: Optional[PollExitResponse]=None, *, printer: Union['PrinterTerm', 'PrinterJupyter']) -> None:
    if not poll_exit_response:
        return
    progress = poll_exit_response.pusher_stats
    done = poll_exit_response.done
    megabyte = wandb.util.POW_2_BYTES[2][1]
    line = f'{progress.uploaded_bytes / megabyte:.3f} MB of {progress.total_bytes / megabyte:.3f} MB uploaded'
    if progress.deduped_bytes > 0:
        line += f' ({progress.deduped_bytes / megabyte:.3f} MB deduped)\r'
    else:
        line += '\r'
    percent_done = 1.0 if progress.total_bytes == 0 else progress.uploaded_bytes / progress.total_bytes
    printer.progress_update(line, percent_done)
    if done:
        printer.progress_close()
        dedupe_fraction = progress.deduped_bytes / float(progress.total_bytes) if progress.total_bytes > 0 else 0
        if dedupe_fraction > 0.01:
            printer.display(f'W&B sync reduced upload amount by {dedupe_fraction * 100:.1f}%             ')