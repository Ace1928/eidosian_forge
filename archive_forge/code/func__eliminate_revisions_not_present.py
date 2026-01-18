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
def _eliminate_revisions_not_present(self, revision_ids):
    """Check every revision id in revision_ids to see if we have it.

        Returns a set of the present revisions.
        """
    with self.lock_read():
        result = []
        graph = self.get_graph()
        parent_map = graph.get_parent_map(revision_ids)
        return list(parent_map)