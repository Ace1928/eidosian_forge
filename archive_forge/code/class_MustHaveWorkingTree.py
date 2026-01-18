from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
class MustHaveWorkingTree(errors.BzrError):
    _fmt = "Branching '%(url)s'(%(format)s) must create a working tree."

    def __init__(self, format, url):
        errors.BzrError.__init__(self, format=format, url=url)