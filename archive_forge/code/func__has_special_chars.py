import contextlib
import os
import platform
import re
import shutil
import tempfile
from typing import Any, Iterator, List, Mapping, Optional, Tuple, Union
from cmdstanpy import _TMPDIR
from .json import write_stan_json
from .logging import get_logger
@classmethod
def _has_special_chars(cls, file_path: str) -> bool:
    if platform.system() == 'Windows':
        return bool(cls.WINDOWS_PATTERN.search(file_path))
    return bool(cls.UNIXISH_PATTERN.search(file_path))