import contextlib
import ctypes
import errno
import logging
import os
import platform
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import IO, Any, BinaryIO, Generator, Optional
from wandb.sdk.lib.paths import StrPath
def _reflink_linux(existing_path: Path, new_path: Path) -> None:
    """Create a reflink to `existing_path` at `new_path` on Linux."""
    import fcntl
    FICLONE = 1074041865
    with open(existing_path, 'rb') as t_f, open(new_path, 'wb+') as l_f:
        fcntl.ioctl(l_f.fileno(), FICLONE, t_f.fileno())