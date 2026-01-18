from typing import Callable, Optional
from . import branch as _mod_branch
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
class DirectoryLookupFailure(errors.BzrError):
    """Base type for lookup errors."""