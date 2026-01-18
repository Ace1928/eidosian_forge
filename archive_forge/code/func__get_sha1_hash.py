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
def _get_sha1_hash(self, file_path: Union[str, Path]) -> str:
    """Return the sha1 hash of a file

        Args:
            file_path (str, Path):
                Path to the file whose sha1 hash is to be calculalated and returned.

        Returns:
            (str):
                The SHA1 computed.
        """
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(self._sha1_buf_size)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()