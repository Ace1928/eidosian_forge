from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
def _get_selected_branch(self):
    """Return the name of the branch selected by the user.

        Returns: Name of the branch selected by the user, or "".
        """
    branch = self.root_transport.get_segment_parameters().get('branch')
    if branch is None:
        branch = ''
    return urlutils.unescape(branch)