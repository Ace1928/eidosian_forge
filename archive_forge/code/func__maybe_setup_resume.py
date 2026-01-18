import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime
from queue import Queue
from typing import (
import requests
import wandb
from wandb import util
from wandb.errors import CommError, UsageError
from wandb.errors.util import ProtobufErrorHandler
from wandb.filesync.dir_watcher import DirWatcher
from wandb.proto import wandb_internal_pb2
from wandb.sdk.artifacts.artifact_saver import ArtifactSaver
from wandb.sdk.interface import interface
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import (
from wandb.sdk.internal.file_pusher import FilePusher
from wandb.sdk.internal.job_builder import JobBuilder
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib import (
from wandb.sdk.lib.mailbox import ContextCancelledError
from wandb.sdk.lib.proto_util import message_to_dict
def _maybe_setup_resume(self, run: 'RunRecord') -> Optional['wandb_internal_pb2.ErrorInfo']:
    """Queries the backend for a run; fail if the settings are incompatible."""
    if not self._settings.resume:
        return None
    entity = run.entity or self._entity
    logger.info('checking resume status for %s/%s/%s', entity, run.project, run.run_id)
    resume_status = self._api.run_resume_status(entity=entity, project_name=run.project, name=run.run_id)
    if not resume_status:
        if self._settings.resume == 'must':
            error = wandb_internal_pb2.ErrorInfo()
            error.code = wandb_internal_pb2.ErrorInfo.ErrorCode.USAGE
            error.message = f"You provided an invalid value for the `resume` argument. The value 'must' is not a valid option for resuming a run ({run.run_id}) that does not exist. Please check your inputs and try again with a valid run ID. If you are trying to start a new run, please omit the `resume` argument or use `resume='allow'`."
            return error
        return None
    if self._settings.resume == 'never':
        error = wandb_internal_pb2.ErrorInfo()
        error.code = wandb_internal_pb2.ErrorInfo.ErrorCode.USAGE
        error.message = f"You provided an invalid value for the `resume` argument. The value 'never' is not a valid option for resuming a run ({run.run_id}) that already exists. Please check your inputs and try again with a valid value for the `resume` argument."
        return error
    history = {}
    events = {}
    config = {}
    summary = {}
    try:
        events_rt = 0
        history_rt = 0
        history = json.loads(resume_status['historyTail'])
        if history:
            history = json.loads(history[-1])
            history_rt = history.get('_runtime', 0)
        events = json.loads(resume_status['eventsTail'])
        if events:
            events = json.loads(events[-1])
            events_rt = events.get('_runtime', 0)
        config = json.loads(resume_status['config'] or '{}')
        summary = json.loads(resume_status['summaryMetrics'] or '{}')
        new_runtime = summary.get('_wandb', {}).get('runtime', None)
        if new_runtime is not None:
            self._resume_state.wandb_runtime = new_runtime
        tags = resume_status.get('tags') or []
    except (IndexError, ValueError) as e:
        logger.error('unable to load resume tails', exc_info=e)
        if self._settings.resume == 'must':
            error = wandb_internal_pb2.ErrorInfo()
            error.code = wandb_internal_pb2.ErrorInfo.ErrorCode.USAGE
            error.message = "resume='must' but could not resume (%s) " % run.run_id
            return error
    self._resume_state.runtime = max(events_rt, history_rt)
    last_step = history.get('_step', 0)
    history_line_count = resume_status['historyLineCount']
    self._resume_state.step = last_step + 1 if history_line_count > 0 else last_step
    self._resume_state.history = history_line_count
    self._resume_state.events = resume_status['eventsLineCount']
    self._resume_state.output = resume_status['logLineCount']
    self._resume_state.config = config
    self._resume_state.summary = summary
    self._resume_state.tags = tags
    self._resume_state.resumed = True
    logger.info('configured resuming with: %s' % self._resume_state)
    return None