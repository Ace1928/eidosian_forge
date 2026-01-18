import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Union
from ..constants import HF_HUB_CACHE
from . import logging
class CacheNotFound(Exception):
    """Exception thrown when the Huggingface cache is not found."""
    cache_dir: Union[str, Path]

    def __init__(self, msg: str, cache_dir: Union[str, Path], *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
        self.cache_dir = cache_dir