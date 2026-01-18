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
def _generate_root_texts(self, revs):
    """This will be called by get_stream between fetching weave texts and
        fetching the inventory weave.
        """
    if self._rich_root_upgrade():
        return _mod_fetch.Inter1and2Helper(self.from_repository).generate_root_texts(revs)
    else:
        return []