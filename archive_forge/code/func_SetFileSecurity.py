from __future__ import annotations
import errno
import os
import site
import stat
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional
import platformdirs
from .utils import deprecation
def SetFileSecurity(lpFileName: Any, RequestedInformation: Any, pSecurityDescriptor: Any) -> Any:
    advapi32.SetFileSecurityW(lpFileName, RequestedInformation, pSecurityDescriptor)