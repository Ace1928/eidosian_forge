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
def _copy_uncompressed(src: Path, dest: Path, thread: Optional[int], blocksize: int) -> None:
    """Helper to copy a file and uncompress it at the same time."""
    with open(str(dest), 'wb') as destf:
        with pgzip.open(str(src), 'rb', thread=thread, blocksize=blocksize) as srcf:
            while True:
                buf = srcf.read(blocksize)
                if len(buf) == 0:
                    break
                destf.write(buf)