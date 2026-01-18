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
def _make_artifact_manifest(self, artifact_manifest: ArtifactManifest, obj: Optional[pb.ArtifactManifest]=None) -> pb.ArtifactManifest:
    proto_manifest = obj or pb.ArtifactManifest()
    proto_manifest.version = artifact_manifest.version()
    proto_manifest.storage_policy = artifact_manifest.storage_policy.name()
    for k, v in artifact_manifest.storage_policy.config().items() or {}.items():
        cfg = proto_manifest.storage_policy_config.add()
        cfg.key = k
        cfg.value_json = json.dumps(v)
    for entry in sorted(artifact_manifest.entries.values(), key=lambda k: k.path):
        proto_entry = proto_manifest.contents.add()
        proto_entry.path = entry.path
        proto_entry.digest = entry.digest
        if entry.size:
            proto_entry.size = entry.size
        if entry.birth_artifact_id:
            proto_entry.birth_artifact_id = entry.birth_artifact_id
        if entry.ref:
            proto_entry.ref = entry.ref
        if entry.local_path:
            proto_entry.local_path = entry.local_path
        for k, v in entry.extra.items():
            proto_extra = proto_entry.extra.add()
            proto_extra.key = k
            proto_extra.value_json = json.dumps(v)
    return proto_manifest