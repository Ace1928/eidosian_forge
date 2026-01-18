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
def _init_run(self, run: 'RunRecord', config_dict: Optional[sender_config.BackendConfigDict]) -> None:
    start_time = run.start_time.ToMicroseconds() / 1000000.0 - self._resume_state.runtime
    if self._resume_state and self._resume_state.tags and (not run.tags):
        run.tags.extend(self._resume_state.tags)
    server_run, inserted, server_messages = self._api.upsert_run(name=run.run_id, entity=run.entity or None, project=run.project or None, group=run.run_group or None, job_type=run.job_type or None, display_name=run.display_name or None, notes=run.notes or None, tags=run.tags[:] or None, config=config_dict or None, sweep_name=run.sweep_id or None, host=run.host or None, program_path=self._settings.program or None, repo=run.git.remote_url or None, commit=run.git.commit or None)
    if run.sweep_id:
        self._job_builder.disable = True
    self._server_messages = server_messages or []
    self._run = run
    if self._resume_state.resumed:
        self._run.resumed = True
        if self._resume_state.wandb_runtime is not None:
            self._run.runtime = self._resume_state.wandb_runtime
    elif not inserted:
        self._telemetry_obj.feature.maybe_run_overwrite = True
    self._run.starting_step = self._resume_state.step
    self._run.start_time.FromMicroseconds(int(start_time * 1000000.0))
    self._run.config.CopyFrom(self._interface._make_config(config_dict))
    if self._resume_state.summary is not None:
        self._run.summary.CopyFrom(self._interface._make_summary_from_dict(self._resume_state.summary))
    storage_id = server_run.get('id')
    if storage_id:
        self._run.storage_id = storage_id
    id = server_run.get('name')
    if id:
        self._api.set_current_run_id(id)
    display_name = server_run.get('displayName')
    if display_name:
        self._run.display_name = display_name
    project = server_run.get('project')
    if project:
        project_name = project.get('name')
        if project_name:
            self._run.project = project_name
            self._project = project_name
            self._api_settings['project'] = project_name
            self._api.set_setting('project', project_name)
        entity = project.get('entity')
        if entity:
            entity_name = entity.get('name')
            if entity_name:
                self._run.entity = entity_name
                self._entity = entity_name
                self._api_settings['entity'] = entity_name
                self._api.set_setting('entity', entity_name)
    sweep_id = server_run.get('sweepName')
    if sweep_id:
        self._run.sweep_id = sweep_id
    if os.getenv('SPELL_RUN_URL'):
        self._sync_spell()