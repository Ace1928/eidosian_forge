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
def _get_simple_inventory_stream(self, revision_ids, missing=False):
    from_weave = self.from_repository.inventories
    if missing:
        delta_closure = True
    else:
        delta_closure = not self.delta_on_metadata()
    yield ('inventories', from_weave.get_record_stream([(rev_id,) for rev_id in revision_ids], self.inventory_fetch_order(), delta_closure))