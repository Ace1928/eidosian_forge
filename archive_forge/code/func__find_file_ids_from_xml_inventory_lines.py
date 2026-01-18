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
def _find_file_ids_from_xml_inventory_lines(self, line_iterator, revision_keys):
    """Helper routine for fileids_altered_by_revision_ids.

        This performs the translation of xml lines to revision ids.

        :param line_iterator: An iterator of lines, origin_version_id
        :param revision_keys: The revision ids to filter for. This should be a
            set or other type which supports efficient __contains__ lookups, as
            the revision key from each parsed line will be looked up in the
            revision_keys filter.
        :return: a dictionary mapping altered file-ids to an iterable of
            revision_ids. Each altered file-ids has the exact revision_ids that
            altered it listed explicitly.
        """
    seen = set(self._serializer._find_text_key_references(line_iterator))
    parent_keys = self._find_parent_keys_of_revisions(revision_keys)
    parent_seen = set(self._serializer._find_text_key_references(self._inventory_xml_lines_for_keys(parent_keys)))
    new_keys = seen - parent_seen
    result = {}
    setdefault = result.setdefault
    for key in new_keys:
        setdefault(key[0], set()).add(key[-1])
    return result