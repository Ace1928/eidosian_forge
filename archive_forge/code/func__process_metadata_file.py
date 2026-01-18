import copy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import torch
from torch import Tensor
from .pygit import PyGit
from .sha1_store import SHA1_Store
def _process_metadata_file(self, metadata_fname: Path) -> Path:
    """Create a metadata_file corresponding to the file to be tracked by weigit if
        the first version of the file is encountered. If a version already exists, open
        the file and get the sha1_hash of the last version as parent_sha1.
        """
    metadata_file = self._dot_wgit_dir_path.joinpath(metadata_fname)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    if not metadata_file.exists() or not metadata_file.stat().st_size:
        metadata_file.touch()
    else:
        with open(metadata_file, 'r') as f:
            ref_data = json.load(f)
    return metadata_file