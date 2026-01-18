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
def _get_tmp_file_path(self) -> Path:
    """Helper to get a tmp file name under self.tmp_dir."""
    fd, name = tempfile.mkstemp(dir=self._tmp_dir)
    os.close(fd)
    return Path(name)