import hashlib
import math
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from urllib.parse import quote
import requests
import urllib3
from wandb.errors.term import termwarn
from wandb.sdk.artifacts.artifact_file_cache import (
from wandb.sdk.artifacts.storage_handlers.azure_handler import AzureHandler
from wandb.sdk.artifacts.storage_handlers.gcs_handler import GCSHandler
from wandb.sdk.artifacts.storage_handlers.http_handler import HTTPHandler
from wandb.sdk.artifacts.storage_handlers.local_file_handler import LocalFileHandler
from wandb.sdk.artifacts.storage_handlers.multi_handler import MultiHandler
from wandb.sdk.artifacts.storage_handlers.s3_handler import S3Handler
from wandb.sdk.artifacts.storage_handlers.tracking_handler import TrackingHandler
from wandb.sdk.artifacts.storage_handlers.wb_artifact_handler import WBArtifactHandler
from wandb.sdk.artifacts.storage_handlers.wb_local_artifact_handler import (
from wandb.sdk.artifacts.storage_layout import StorageLayout
from wandb.sdk.artifacts.storage_policies.register import WANDB_STORAGE_POLICY
from wandb.sdk.artifacts.storage_policy import StoragePolicy
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib.hashutil import B64MD5, b64_to_hex_id, hex_to_b64_id
from wandb.sdk.lib.paths import FilePathStr, URIStr
def _file_url(self, api: InternalApi, entity_name: str, manifest_entry: 'ArtifactManifestEntry') -> str:
    storage_layout = self._config.get('storageLayout', StorageLayout.V1)
    storage_region = self._config.get('storageRegion', 'default')
    md5_hex = b64_to_hex_id(B64MD5(manifest_entry.digest))
    if storage_layout == StorageLayout.V1:
        return '{}/artifacts/{}/{}'.format(api.settings('base_url'), entity_name, md5_hex)
    elif storage_layout == StorageLayout.V2:
        return '{}/artifactsV2/{}/{}/{}/{}'.format(api.settings('base_url'), storage_region, entity_name, quote(manifest_entry.birth_artifact_id if manifest_entry.birth_artifact_id is not None else ''), md5_hex)
    else:
        raise Exception(f'unrecognized storage layout: {storage_layout}')