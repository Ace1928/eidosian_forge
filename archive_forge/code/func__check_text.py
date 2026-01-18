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
def _check_text(self, record, checker, item_data):
    """Check a single text."""
    chunks = record.get_bytes_as('chunked')
    sha1 = osutils.sha_strings(chunks)
    length = sum(map(len, chunks))
    if item_data and sha1 != item_data[1]:
        checker._report_items.append('sha1 mismatch: %s has sha1 %s expected %s referenced by %s' % (record.key, sha1, item_data[1], item_data[2]))