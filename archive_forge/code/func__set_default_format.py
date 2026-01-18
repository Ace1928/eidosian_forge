from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
@classmethod
def _set_default_format(klass, format):
    """Set default format (for testing behavior of defaults only)"""
    klass._default_format = format