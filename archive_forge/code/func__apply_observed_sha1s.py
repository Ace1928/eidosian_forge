import contextlib
import errno
import os
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from .. import (annotate, conflicts, controldir, errors, lock, multiparent,
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, ui, urlutils
from ..filters import ContentFilterContext, filtered_output_bytes
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..progress import ProgressPhase
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import find_previous_path
from . import inventory, inventorytree
from .conflicts import Conflict
def _apply_observed_sha1s(self):
    """After we have finished renaming everything, update observed sha1s

        This has to be done after self._tree.apply_inventory_delta, otherwise
        it doesn't know anything about the files we are updating. Also, we want
        to do this as late as possible, so that most entries end up cached.
        """
    paths = FinalPaths(self)
    for trans_id, observed in self._observed_sha1s.items():
        path = paths.get_path(trans_id)
        self._tree._observed_sha1(path, observed)