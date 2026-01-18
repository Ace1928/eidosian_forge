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
def _add_file_to_weave(self, file_id, fileobj, parents, nostore_sha, size):
    parent_keys = tuple([(file_id, parent) for parent in parents])
    return self.repository.texts.add_content(versionedfile.FileContentFactory((file_id, self._new_revision_id), parent_keys, fileobj, size=size), nostore_sha=nostore_sha, random_id=self.random_revid)[0:2]