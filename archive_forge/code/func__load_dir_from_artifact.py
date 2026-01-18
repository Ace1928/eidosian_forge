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
def _load_dir_from_artifact(source_artifact: 'Artifact', path: str) -> str:
    dl_path = None
    for p, _ in source_artifact.manifest.entries.items():
        if p.startswith(path):
            example_path = source_artifact.get_entry(p).download()
            if dl_path is None:
                root = example_path[:-len(p)]
                dl_path = os.path.join(root, path)
    assert dl_path is not None, f'Could not find directory {path} in artifact'
    return dl_path