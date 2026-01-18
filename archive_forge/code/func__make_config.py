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
def _make_config(self, data: Optional[dict]=None, key: Optional[Union[Tuple[str, ...], str]]=None, val: Optional[Any]=None, obj: Optional[pb.ConfigRecord]=None) -> pb.ConfigRecord:
    config = obj or pb.ConfigRecord()
    if data:
        for k, v in data.items():
            update = config.update.add()
            update.key = k
            update.value_json = json_dumps_safer(json_friendly(v)[0])
    if key:
        update = config.update.add()
        if isinstance(key, tuple):
            for k in key:
                update.nested_key.append(k)
        else:
            update.key = key
        update.value_json = json_dumps_safer(json_friendly(val)[0])
    return config