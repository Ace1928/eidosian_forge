import copy
import json
import logging
import os
import platform
import sys
import tempfile
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
import wandb
import wandb.env
from wandb import trigger
from wandb.errors import CommError, Error, UsageError
from wandb.errors.util import ProtobufErrorHandler
from wandb.integration import sagemaker
from wandb.integration.magic import magic_install
from wandb.sdk.lib import runid
from wandb.sdk.lib.paths import StrPath
from wandb.util import _is_artifact_representation
from . import wandb_login, wandb_setup
from .backend.backend import Backend
from .lib import (
from .lib.deprecate import Deprecated, deprecate
from .lib.mailbox import Mailbox, MailboxProgress
from .lib.printer import Printer, get_printer
from .lib.wburls import wburls
from .wandb_helper import parse_config
from .wandb_run import Run, TeardownHook, TeardownStage
from .wandb_settings import Settings, Source
def _make_run_disabled(self) -> RunDisabled:
    drun = RunDisabled()
    drun.config = wandb.wandb_sdk.wandb_config.Config()
    drun.config.update(self.sweep_config)
    drun.config.update(self.config)
    drun.summary = SummaryDisabled()
    drun.log = lambda data, *_, **__: drun.summary.update(data)
    drun.finish = lambda *_, **__: module.unset_globals()
    drun.step = 0
    drun.resumed = False
    drun.disabled = True
    drun.id = runid.generate_id()
    drun.name = 'dummy-' + drun.id
    drun.dir = tempfile.gettempdir()
    module.set_global(run=drun, config=drun.config, log=drun.log, summary=drun.summary, save=drun.save, use_artifact=drun.use_artifact, log_artifact=drun.log_artifact, define_metric=drun.define_metric, plot_table=drun.plot_table, alert=drun.alert)
    return drun