from io import BytesIO
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import serializer
from breezy.i18n import gettext
from breezy.bzr.testament import Testament
from .. import errors
from ..decorators import only_raises
from ..repository import (CommitBuilder, FetchResult, InterRepository,
from ..trace import mutter, note
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import InventoryTreeChange
from .repository import MetaDirRepository, RepositoryFormatMetaDir
def _get_delta(self, ie, basis_inv, path):
    """Get a delta against the basis inventory for ie."""
    if not basis_inv.has_id(ie.file_id):
        result = (None, path, ie.file_id, ie)
        self._basis_delta.append(result)
        return result
    elif ie != basis_inv.get_entry(ie.file_id):
        result = (basis_inv.id2path(ie.file_id), path, ie.file_id, ie)
        self._basis_delta.append(result)
        return result
    else:
        return None