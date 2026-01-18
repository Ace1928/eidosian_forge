import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Union
from ..constants import HF_HUB_CACHE
from . import logging
class CorruptedCacheException(Exception):
    """Exception for any unexpected structure in the Huggingface cache-system."""