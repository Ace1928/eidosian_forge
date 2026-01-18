from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
def _get_all_lazy(self):
    """Return getters for all formats, even those not usable in metadirs."""
    result = [self._dict[name].get_obj for name in self.keys()]
    result.extend(self._get_extra())
    return result