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
def _make_summary(self, summary_record: sr.SummaryRecord) -> pb.SummaryRecord:
    pb_summary_record = pb.SummaryRecord()
    for item in summary_record.update:
        pb_summary_item = pb_summary_record.update.add()
        key_length = len(item.key)
        assert key_length > 0
        if key_length > 1:
            pb_summary_item.nested_key.extend(item.key)
        else:
            pb_summary_item.key = item.key[0]
        path_from_root = '.'.join(item.key)
        json_value = self._summary_encode(item.value, path_from_root)
        json_value, _ = json_friendly(json_value)
        pb_summary_item.value_json = json.dumps(json_value, cls=WandBJSONEncoderOld)
    for item in summary_record.remove:
        pb_summary_item = pb_summary_record.remove.add()
        key_length = len(item.key)
        assert key_length > 0
        if key_length > 1:
            pb_summary_item.nested_key.extend(item.key)
        else:
            pb_summary_item.key = item.key[0]
    return pb_summary_record