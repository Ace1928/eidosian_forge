import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Union, cast
import pgzip
import torch
from torch import Tensor
from fairscale.internal.containers import from_np, to_np
from .utils import ExitCode
def _add_ref(self, current_sha1_hash: str, inc: bool, compressed: bool) -> int:
    """
        Update the reference count.

        If the reference counting file does not have this sha1, then a new tracking
        entry of the added.

        Args:
            current_sha1_hash (str):
                The sha1 hash of the incoming added file.
            inc (bool):
                Increment or decrement.

        Returns:
            (int):
                Resulting ref count.
        """
    if current_sha1_hash not in self._json_dict:
        entry = {}
    else:
        entry = self._json_dict[current_sha1_hash]
    entry = _get_json_entry(entry)
    entry[ENTRY_RF_KEY] += 1 if inc else -1
    assert entry[ENTRY_RF_KEY] >= 0, 'negative ref count'
    entry[ENTRY_COMP_KEY] = compressed
    self._json_dict[current_sha1_hash] = entry
    return entry[ENTRY_RF_KEY]