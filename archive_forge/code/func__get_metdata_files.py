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
def _get_metdata_files(self) -> Dict[str, bool]:
    """Walk the directories that contain the metadata files and check the
        status of those files, whether they have been modified or not.

        Dict[str, bool] is a path in string and whether the file is_modified.
        """
    metadata_d = dict()
    for file in self._dot_wgit_dir_path.iterdir():
        if file.name not in {'sha1_store', '.git', '.gitignore'}:
            for path in file.rglob('*'):
                if path.is_file():
                    rel_path = str(path.relative_to(self._dot_wgit_dir_path))
                    metadata_d[rel_path] = self._is_file_modified(path)
    return metadata_d