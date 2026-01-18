import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
from .folder import ImageFolder
from .utils import check_integrity, extract_archive, verify_str_arg
def _verify_archive(root: str, file: str, md5: str) -> None:
    if not check_integrity(os.path.join(root, file), md5):
        msg = 'The archive {} is not present in the root directory or is corrupted. You need to download it externally and place it in {}.'
        raise RuntimeError(msg.format(file, root))