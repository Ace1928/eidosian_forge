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
def _is_file_modified(self, file: Path) -> bool:
    """Checks whether a file has been modified since its last recorded modification
        time recorded in the metadata_file.
        """
    with open(file) as f:
        data = json.load(f)
    last_mod_timestamp = data[LAST_MODIFIED_TS_KEY]
    curr_mod_timestamp = Path(data[REL_PATH_KEY]).stat().st_mtime
    return not curr_mod_timestamp == last_mod_timestamp