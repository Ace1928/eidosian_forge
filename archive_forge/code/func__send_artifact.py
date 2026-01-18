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
def _send_artifact(self, artifact: 'ArtifactRecord', history_step: Optional[int]=None) -> Optional[Dict]:
    from wandb.util import parse_version
    assert self._pusher
    saver = ArtifactSaver(api=self._api, digest=artifact.digest, manifest_json=_manifest_json_from_proto(artifact.manifest), file_pusher=self._pusher, is_user_created=artifact.user_created)
    if artifact.distributed_id:
        max_cli_version = self._max_cli_version()
        if max_cli_version is None or parse_version(max_cli_version) < parse_version('0.10.16'):
            logger.warning("This W&B Server doesn't support distributed artifacts, have your administrator install wandb/local >= 0.9.37")
            return None
    metadata = json.loads(artifact.metadata) if artifact.metadata else None
    res = saver.save(type=artifact.type, name=artifact.name, client_id=artifact.client_id, sequence_client_id=artifact.sequence_client_id, metadata=metadata, ttl_duration_seconds=artifact.ttl_duration_seconds or None, description=artifact.description or None, aliases=artifact.aliases, use_after_commit=artifact.use_after_commit, distributed_id=artifact.distributed_id, finalize=artifact.finalize, incremental=artifact.incremental_beta1, history_step=history_step, base_id=artifact.base_id or None)
    self._job_builder._handle_server_artifact(res, artifact)
    return res