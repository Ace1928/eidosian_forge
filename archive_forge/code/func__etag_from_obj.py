import os
import time
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qsl, urlparse
from wandb import util
from wandb.errors import CommError
from wandb.errors.term import termlog
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_handler import DEFAULT_MAX_OBJECTS, StorageHandler
from wandb.sdk.lib.hashutil import ETag
from wandb.sdk.lib.paths import FilePathStr, StrPath, URIStr
@staticmethod
def _etag_from_obj(obj: Union['boto3.s3.Object', 'boto3.s3.ObjectSummary']) -> ETag:
    etag: ETag
    etag = obj.e_tag[1:-1]
    return etag