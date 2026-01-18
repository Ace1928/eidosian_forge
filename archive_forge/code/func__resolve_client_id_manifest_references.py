import concurrent.futures
import json
import os
import sys
import tempfile
from typing import TYPE_CHECKING, Awaitable, Dict, Optional, Sequence
import wandb
import wandb.filesync.step_prepare
from wandb import util
from wandb.sdk.artifacts.artifact_manifest import ArtifactManifest
from wandb.sdk.artifacts.staging import get_staging_dir
from wandb.sdk.lib.hashutil import B64MD5, b64_to_hex_id, md5_file_b64
from wandb.sdk.lib.paths import URIStr
def _resolve_client_id_manifest_references(self) -> None:
    for entry_path in self._manifest.entries:
        entry = self._manifest.entries[entry_path]
        if entry.ref is not None:
            if entry.ref.startswith('wandb-client-artifact:'):
                client_id = util.host_from_path(entry.ref)
                artifact_file_path = util.uri_from_path(entry.ref)
                artifact_id = self._api._resolve_client_id(client_id)
                if artifact_id is None:
                    raise RuntimeError(f'Could not resolve client id {client_id}')
                entry.ref = URIStr('wandb-artifact://{}/{}'.format(b64_to_hex_id(B64MD5(artifact_id)), artifact_file_path))