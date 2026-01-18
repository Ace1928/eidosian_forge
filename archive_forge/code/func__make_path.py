import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
def _make_path(self, filename) -> str:
    return os.path.join(self.cache_dir, filename)