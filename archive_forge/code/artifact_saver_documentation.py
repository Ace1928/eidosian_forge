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
Remove all staging copies of local files.

        We made a staging copy of each local file to freeze it at "add" time.
        We need to delete them once we've uploaded the file or confirmed we
        already have a committed copy.
        