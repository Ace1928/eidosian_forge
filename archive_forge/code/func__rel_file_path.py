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
def _rel_file_path(self, filepath: Path) -> Path:
    """Find the relative part to the filepath from the current working
        directory and return the relative path.
        """
    filepath = filepath.resolve()
    for i, (x, y) in enumerate(zip(filepath.parts, Path.cwd().parts)):
        pass
    return Path(*filepath.parts[i:])