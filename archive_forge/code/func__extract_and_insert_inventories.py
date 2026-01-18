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
def _extract_and_insert_inventories(self, substream, serializer, parse_delta=None):
    """Generate a new inventory versionedfile in target, converting data.

        The inventory is retrieved from the source, (deserializing it), and
        stored in the target (reserializing it in a different format).
        """
    target_rich_root = self.target_repo._format.rich_root_data
    target_tree_refs = self.target_repo._format.supports_tree_reference
    for record in substream:
        lines = record.get_bytes_as('lines')
        revision_id = record.key[0]
        inv = serializer.read_inventory_from_lines(lines, revision_id)
        parents = [key[0] for key in record.parents]
        self.target_repo.add_inventory(revision_id, inv, parents)
        del inv