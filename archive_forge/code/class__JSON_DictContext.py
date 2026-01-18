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
class _JSON_DictContext:
    """Helper class that handles syncing of a json and a dict."""

    def __init__(self, s: 'SHA1_Store', readonly: bool) -> None:
        self._s = s
        self._readonly = readonly

    def __enter__(self) -> None:
        """Load from file."""
        assert self._s._json_dict is None
        if self._s._metadata_file_path.exists():
            with open(self._s._metadata_file_path, 'r') as f:
                self._s._json_dict = json.load(f)
        else:
            self._s._json_dict = {}

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        """Store back to file."""
        assert isinstance(self._s._json_dict, dict)
        if not self._readonly:
            with open(self._s._metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(self._s._json_dict, f, ensure_ascii=False, indent=2)
        self._s._json_dict = None