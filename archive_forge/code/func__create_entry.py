from pathlib import PurePosixPath
from types import ModuleType
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union
from urllib.parse import ParseResult, parse_qsl, urlparse
import wandb
from wandb import util
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_handler import DEFAULT_MAX_OBJECTS, StorageHandler
from wandb.sdk.lib.hashutil import ETag
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath, URIStr
def _create_entry(self, blob_properties: 'azure.storage.blob.BlobProperties', path: StrPath, ref: URIStr) -> ArtifactManifestEntry:
    extra = {'etag': blob_properties.etag.strip('"')}
    if blob_properties.version_id:
        extra['versionID'] = blob_properties.version_id
    return ArtifactManifestEntry(path=path, ref=ref, digest=blob_properties.etag.strip('"'), size=blob_properties.size, extra=extra)