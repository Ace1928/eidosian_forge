import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Union
from ..constants import HF_HUB_CACHE
from . import logging
def _try_delete_path(path: Path, path_type: str) -> None:
    """Try to delete a local file or folder.

    If the path does not exists, error is logged as a warning and then ignored.

    Args:
        path (`Path`)
            Path to delete. Can be a file or a folder.
        path_type (`str`)
            What path are we deleting ? Only for logging purposes. Example: "snapshot".
    """
    logger.info(f'Delete {path_type}: {path}')
    try:
        if path.is_file():
            os.remove(path)
        else:
            shutil.rmtree(path)
    except FileNotFoundError:
        logger.warning(f"Couldn't delete {path_type}: file not found ({path})", exc_info=True)
    except PermissionError:
        logger.warning(f"Couldn't delete {path_type}: permission denied ({path})", exc_info=True)