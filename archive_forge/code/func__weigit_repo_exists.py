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
def _weigit_repo_exists(self, check_dir: Path) -> bool:
    """Returns True if a valid WeiGit repo exists in the path: check_dir."""
    wgit_exists, git_exists, gitignore_exists = self._weigit_repo_file_check(check_dir)
    return wgit_exists and git_exists and gitignore_exists