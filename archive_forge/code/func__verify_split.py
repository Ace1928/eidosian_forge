import os
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin
from .folder import default_loader
from .utils import check_integrity, download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
def _verify_split(self, split: str) -> str:
    return verify_str_arg(split, 'split', self._SPLITS)