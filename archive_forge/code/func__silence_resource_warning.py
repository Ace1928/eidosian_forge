from __future__ import annotations
import os
import subprocess
import sys
import warnings
from typing import Any, Optional, Sequence
def _silence_resource_warning(popen: Optional[subprocess.Popen[Any]]) -> None:
    """Silence Popen's ResourceWarning.

    Note this should only be used if the process was created as a daemon.
    """
    if popen is not None:
        popen.returncode = 0