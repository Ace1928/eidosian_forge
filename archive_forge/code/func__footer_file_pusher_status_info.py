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
def _footer_file_pusher_status_info(poll_exit_responses: Optional[Union[PollExitResponse, List[Optional[PollExitResponse]]]]=None, *, printer: Union['PrinterTerm', 'PrinterJupyter']) -> None:
    if not poll_exit_responses:
        return
    if isinstance(poll_exit_responses, PollExitResponse):
        Run._footer_single_run_file_pusher_status_info(poll_exit_responses, printer=printer)
    elif isinstance(poll_exit_responses, list):
        poll_exit_responses_list = poll_exit_responses
        assert all((response is None or isinstance(response, PollExitResponse) for response in poll_exit_responses_list))
        if len(poll_exit_responses_list) == 0:
            return
        elif len(poll_exit_responses_list) == 1:
            Run._footer_single_run_file_pusher_status_info(poll_exit_responses_list[0], printer=printer)
        else:
            Run._footer_multiple_runs_file_pusher_status_info(poll_exit_responses_list, printer=printer)
    else:
        logger.error(f'Got the type `{type(poll_exit_responses)}` for `poll_exit_responses`. Expected either None, PollExitResponse or a List[Union[PollExitResponse, None]]')