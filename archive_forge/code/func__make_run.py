import logging
import os
import sys
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, NewType, Optional, Tuple, Union
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_telemetry_pb2 as tpb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.artifacts.artifact_manifest import ArtifactManifest
from wandb.sdk.artifacts.staging import get_staging_dir
from wandb.sdk.lib import json_util as json
from wandb.util import (
from ..data_types.utils import history_dict_to_json, val_to_json
from ..lib.mailbox import MailboxHandle
from . import summary_record as sr
from .message_future import MessageFuture
def _make_run(self, run: 'Run') -> pb.RunRecord:
    proto_run = pb.RunRecord()
    run._make_proto_run(proto_run)
    if run._settings.host:
        proto_run.host = run._settings.host
    if run._config is not None:
        config_dict = run._config._as_dict()
        self._make_config(data=config_dict, obj=proto_run.config)
    if run._telemetry_obj:
        proto_run.telemetry.MergeFrom(run._telemetry_obj)
    return proto_run