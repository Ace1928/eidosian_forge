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
def _recursive_search_and_may_init_dot_wgit_dir_path(self, check_dir: Path) -> bool:
    """Search for a wgit repo top level dir from potentiall a subdir of a repo.

            This may set the self._dot_wgit_dir_path if a repo is found.

        Args:
           check_dir (Path):
               Path to the directory from where search is started.

        Returns:
           Returns True if a repo is found.
        """
    assert self._dot_wgit_dir_path is None, f'_dot_wgit_dir_path is already set to {self._dot_wgit_dir_path}'
    if self._weigit_repo_exists(check_dir):
        self._dot_wgit_dir_path = check_dir.joinpath('.wgit')
    else:
        root = Path(check_dir.parts[0])
        while check_dir != root:
            check_dir = check_dir.parent
            if self._weigit_repo_exists(check_dir):
                self._dot_wgit_dir_path = check_dir.joinpath('.wgit')
                break
    return True if self._dot_wgit_dir_path is not None else False