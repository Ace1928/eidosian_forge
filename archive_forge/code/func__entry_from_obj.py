import time
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union
from urllib.parse import ParseResult, urlparse
from wandb import util
from wandb.errors.term import termlog
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_handler import DEFAULT_MAX_OBJECTS, StorageHandler
from wandb.sdk.lib.hashutil import B64MD5
from wandb.sdk.lib.paths import FilePathStr, StrPath, URIStr
def _entry_from_obj(self, obj: 'gcs_module.blob.Blob', path: str, name: Optional[StrPath]=None, prefix: str='', multi: bool=False) -> ArtifactManifestEntry:
    """Create an ArtifactManifestEntry from a GCS object.

        Arguments:
            obj: The GCS object
            path: The GCS-style path (e.g.: "gs://bucket/file.txt")
            name: The user assigned name, or None if not specified
            prefix: The prefix to add (will be the same as `path` for directories)
            multi: Whether or not this is a multi-object add.
        """
    bucket, key, _ = self._parse_uri(path)
    posix_key = PurePosixPath(obj.name)
    posix_path = PurePosixPath(bucket) / PurePosixPath(key)
    posix_prefix = PurePosixPath(prefix)
    posix_name = PurePosixPath(name or '')
    posix_ref = posix_path
    if name is None:
        if str(posix_prefix) in str(posix_key) and posix_prefix != posix_key:
            posix_name = posix_key.relative_to(posix_prefix)
            posix_ref = posix_path / posix_name
        else:
            posix_name = PurePosixPath(posix_key.name)
            posix_ref = posix_path
    elif multi:
        relpath = posix_key.relative_to(posix_prefix)
        posix_name = posix_name / relpath
        posix_ref = posix_path / relpath
    return ArtifactManifestEntry(path=posix_name, ref=URIStr(f'{self._scheme}://{str(posix_ref)}'), digest=obj.md5_hash, size=obj.size, extra=self._extra_from_obj(obj))