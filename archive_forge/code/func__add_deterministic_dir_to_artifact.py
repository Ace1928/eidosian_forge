import os
import shutil
import sys
from typing import (
import wandb
from wandb import util
from wandb.sdk.lib import runid
from wandb.sdk.lib.hashutil import md5_file_hex
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.wb_value import WBValue
def _add_deterministic_dir_to_artifact(artifact: 'Artifact', dir_name: str, target_dir_root: str) -> str:
    file_paths = []
    for dirpath, _, filenames in os.walk(dir_name, topdown=True):
        for fn in filenames:
            file_paths.append(os.path.join(dirpath, fn))
    dirname = md5_file_hex(*file_paths)[:20]
    target_path = LogicalPath(os.path.join(target_dir_root, dirname))
    artifact.add_dir(dir_name, target_path)
    return target_path