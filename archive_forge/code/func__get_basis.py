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
def _get_basis(self, first_revision_id):
    """Get a revision and tree which exists in the target.

        This assumes that first_revision_id is selected for transmission
        because all other ancestors are already present. If we can't find an
        ancestor we fall back to NULL_REVISION since we know that is safe.

        :return: (basis_id, basis_tree)
        """
    first_rev = self.source.get_revision(first_revision_id)
    try:
        basis_id = first_rev.parent_ids[0]
        self.target.get_revision(basis_id)
        basis_tree = self.source.revision_tree(basis_id)
    except (IndexError, errors.NoSuchRevision):
        basis_id = _mod_revision.NULL_REVISION
        basis_tree = self.source.revision_tree(basis_id)
    return (basis_id, basis_tree)