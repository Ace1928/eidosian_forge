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
def _redirect(self, stdout_slave_fd: Optional[int], stderr_slave_fd: Optional[int], console: Optional[str]=None) -> None:
    if console is None:
        console = self._settings.console
    if console == 'wrap':
        if not self._settings._disable_service:
            console = 'wrap_raw'
        else:
            console = 'wrap_emu'
    logger.info('redirect: %s', console)
    out_redir: redirect.RedirectBase
    err_redir: redirect.RedirectBase
    if console in {'redirect', 'wrap_emu'}:
        output_log_path = os.path.join(self._settings.files_dir, filenames.OUTPUT_FNAME)
        if not self._output_writer:
            self._output_writer = filesystem.CRDedupedFile(open(output_log_path, 'wb'))
    if console == 'redirect':
        logger.info('Redirecting console.')
        out_redir = redirect.Redirect(src='stdout', cbs=[lambda data: self._console_callback('stdout', data), self._output_writer.write])
        err_redir = redirect.Redirect(src='stderr', cbs=[lambda data: self._console_callback('stderr', data), self._output_writer.write])
        if os.name == 'nt':

            def wrap_fallback() -> None:
                if self._out_redir:
                    self._out_redir.uninstall()
                if self._err_redir:
                    self._err_redir.uninstall()
                msg = 'Tensorflow detected. Stream redirection is not supported on Windows when tensorflow is imported. Falling back to wrapping stdout/err.'
                wandb.termlog(msg)
                self._redirect(None, None, console='wrap')
            add_import_hook('tensorflow', wrap_fallback)
    elif console == 'wrap_emu':
        logger.info('Wrapping output streams.')
        out_redir = redirect.StreamWrapper(src='stdout', cbs=[lambda data: self._console_callback('stdout', data), self._output_writer.write])
        err_redir = redirect.StreamWrapper(src='stderr', cbs=[lambda data: self._console_callback('stderr', data), self._output_writer.write])
    elif console == 'wrap_raw':
        logger.info('Wrapping output streams.')
        out_redir = redirect.StreamRawWrapper(src='stdout', cbs=[lambda data: self._console_raw_callback('stdout', data)])
        err_redir = redirect.StreamRawWrapper(src='stderr', cbs=[lambda data: self._console_raw_callback('stderr', data)])
    elif console == 'off':
        return
    else:
        raise ValueError('unhandled console')
    try:
        out_redir.save()
        err_redir.save()
        out_redir.install()
        err_redir.install()
        self._out_redir = out_redir
        self._err_redir = err_redir
        logger.info('Redirects installed.')
    except Exception as e:
        print(e)
        logger.error('Failed to redirect.', exc_info=e)
    return