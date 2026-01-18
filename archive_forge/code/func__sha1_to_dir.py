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
def _sha1_to_dir(self, sha1: str) -> Path:
    """Helper to get the internal dir for a file based on its SHA1"""
    assert len(sha1) > 4, 'sha1 too short'
    part1, part2 = (sha1[:2], sha1[2:4])
    return self._path.joinpath(part1, part2)