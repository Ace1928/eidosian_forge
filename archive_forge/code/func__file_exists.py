import os
import contextlib
import json
import shutil
import pathlib
from typing import Any, List
import uuid
from ray.workflow.storage.base import Storage, KeyNotFoundError
import ray.cloudpickle
def _file_exists(path: pathlib.Path) -> bool:
    """During atomic writing, we backup the original file. If the writing
    failed during the middle, then only the backup exists. We consider the
    file exists if the file or the backup file exists. We also automatically
    restore the backup file to the original path if only backup file exists.

    Args:
        path: File path.

    Returns:
        True if the file and backup exists.
    """
    backup_path = path.with_name(f'.{path.name}.backup')
    if path.exists():
        return True
    elif backup_path.exists():
        backup_path.rename(path)
        return True
    return False