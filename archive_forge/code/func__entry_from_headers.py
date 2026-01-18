import os
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union
from urllib.parse import ParseResult
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_handler import StorageHandler
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib.hashutil import ETag
from wandb.sdk.lib.paths import FilePathStr, StrPath, URIStr
def _entry_from_headers(self, headers: 'requests.structures.CaseInsensitiveDict') -> Tuple[Optional[ETag], Optional[int], Dict[str, str]]:
    response_headers = {k.lower(): v for k, v in headers.items()}
    size = None
    if response_headers.get('content-length', None):
        size = int(response_headers['content-length'])
    digest = response_headers.get('etag', None)
    extra = {}
    if digest:
        extra['etag'] = digest
    if digest and digest[:1] == '"' and (digest[-1:] == '"'):
        digest = digest[1:-1]
    return (digest, size, extra)