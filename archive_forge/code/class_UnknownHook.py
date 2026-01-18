from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
class UnknownHook(errors.BzrError):
    _fmt = "The %(type)s hook '%(hook)s' is unknown in this version of breezy."

    def __init__(self, hook_type, hook_name):
        errors.BzrError.__init__(self)
        self.type = hook_type
        self.hook = hook_name