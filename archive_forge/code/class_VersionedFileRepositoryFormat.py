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
class VersionedFileRepositoryFormat(RepositoryFormat):
    """Base class for all repository formats that are VersionedFiles-based."""
    supports_full_versioned_files = True
    supports_versioned_directories = True
    supports_unreferenced_revisions = True
    _commit_inv_deltas = True
    _fetch_order = 'unordered'
    _fetch_uses_deltas = False