from typing import Callable, Optional
from . import branch as _mod_branch
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
class UnsetLocationAlias(DirectoryLookupFailure):
    _fmt = 'No %(alias_name)s location assigned.'

    def __init__(self, alias_name):
        DirectoryLookupFailure.__init__(self, alias_name=alias_name[1:])