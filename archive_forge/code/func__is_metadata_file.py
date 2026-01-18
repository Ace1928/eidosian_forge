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
def _is_metadata_file(self, file: Path) -> bool:
    """Checks whether a file is a valid metadata file by matching keys and
        checking if it has valid json data.
        """
    try:
        with open(file) as f:
            metadata = json.load(f)
        is_metadata = set(metadata.keys()) == {SHA1_KEY, LAST_MODIFIED_TS_KEY, REL_PATH_KEY}
    except json.JSONDecodeError:
        return False
    return is_metadata