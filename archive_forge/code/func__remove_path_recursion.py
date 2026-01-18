import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from . import opt_dry_run, opt_quiet, QTPATHS_CMD, PROJECT_FILE_SUFFIX
def _remove_path_recursion(path: Path):
    """Recursion to remove a file or directory."""
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        for item in path.iterdir():
            _remove_path_recursion(item)
        path.rmdir()