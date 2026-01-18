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
def _find_non_file_keys_to_fetch(self, revision_ids):
    yield ('inventory', None, revision_ids)
    revisions_with_signatures = set(self.signatures.get_parent_map([(r,) for r in revision_ids]))
    revisions_with_signatures = {r for r, in revisions_with_signatures}
    revisions_with_signatures.intersection_update(revision_ids)
    yield ('signatures', None, revisions_with_signatures)
    yield ('revisions', None, revision_ids)