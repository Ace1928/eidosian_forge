import atexit
import concurrent.futures
import contextlib
import json
import multiprocessing.dummy
import os
import re
import shutil
import tempfile
import time
from copy import copy
from datetime import datetime, timedelta
from functools import partial
from pathlib import PurePosixPath
from typing import (
from urllib.parse import urlparse
import requests
import wandb
from wandb import data_types, env, util
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.public import ArtifactCollection, ArtifactFiles, RetryingClient, Run
from wandb.data_types import WBValue
from wandb.errors.term import termerror, termlog, termwarn
from wandb.sdk.artifacts.artifact_download_logger import ArtifactDownloadLogger
from wandb.sdk.artifacts.artifact_instance_cache import artifact_instance_cache
from wandb.sdk.artifacts.artifact_manifest import ArtifactManifest
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.artifact_manifests.artifact_manifest_v1 import (
from wandb.sdk.artifacts.artifact_state import ArtifactState
from wandb.sdk.artifacts.artifact_ttl import ArtifactTTL
from wandb.sdk.artifacts.exceptions import (
from wandb.sdk.artifacts.staging import get_staging_dir
from wandb.sdk.artifacts.storage_layout import StorageLayout
from wandb.sdk.artifacts.storage_policies import WANDB_STORAGE_POLICY
from wandb.sdk.artifacts.storage_policy import StoragePolicy
from wandb.sdk.data_types._dtypes import Type as WBType
from wandb.sdk.data_types._dtypes import TypeRegistry
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib import filesystem, retry, runid, telemetry
from wandb.sdk.lib.hashutil import B64MD5, b64_to_hex_id, md5_file_b64
from wandb.sdk.lib.mailbox import Mailbox
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath, URIStr
from wandb.sdk.lib.runid import generate_id
from wandb.util import get_core_path
from wandb_gql import gql  # noqa: E402
def add_dir(self, local_path: str, name: Optional[str]=None) -> None:
    """Add a local directory to the artifact.

        Arguments:
            local_path: The path of the local directory.
            name: The subdirectory name within an artifact. The name you specify appears
                in the W&B App UI nested by artifact's `type`.
                Defaults to the root of the artifact.

        Raises:
            ArtifactFinalizedError: You cannot make changes to the current artifact
            version because it is finalized. Log a new artifact version instead.
        """
    self._ensure_can_add()
    if not os.path.isdir(local_path):
        raise ValueError('Path is not a directory: %s' % local_path)
    termlog('Adding directory to artifact (%s)... ' % os.path.join('.', os.path.normpath(local_path)), newline=False)
    start_time = time.time()
    paths = []
    for dirpath, _, filenames in os.walk(local_path, followlinks=True):
        for fname in filenames:
            physical_path = os.path.join(dirpath, fname)
            logical_path = os.path.relpath(physical_path, start=local_path)
            if name is not None:
                logical_path = os.path.join(name, logical_path)
            paths.append((logical_path, physical_path))

    def add_manifest_file(log_phy_path: Tuple[str, str]) -> None:
        logical_path, physical_path = log_phy_path
        self._add_local_file(logical_path, physical_path)
    num_threads = 8
    pool = multiprocessing.dummy.Pool(num_threads)
    pool.map(add_manifest_file, paths)
    pool.close()
    pool.join()
    termlog('Done. %.1fs' % (time.time() - start_time), prefix=False)