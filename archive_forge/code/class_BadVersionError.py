import os
from packaging.version import Version
from .config import ET_PROJECTS
class BadVersionError(RuntimeError):
    """Local version is known to contain a critical bug etc."""
    pass