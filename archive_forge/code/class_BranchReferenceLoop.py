from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
class BranchReferenceLoop(errors.BzrError):
    _fmt = 'Can not create branch reference that points at branch itself.'

    def __init__(self, branch):
        errors.BzrError.__init__(self, branch=branch)