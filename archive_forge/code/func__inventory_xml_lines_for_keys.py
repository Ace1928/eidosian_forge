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
def _inventory_xml_lines_for_keys(self, keys):
    """Get a line iterator of the sort needed for findind references.

        Not relevant for non-xml inventory repositories.

        Ghosts in revision_keys are ignored.

        :param revision_keys: The revision keys for the inventories to inspect.
        :return: An iterator over (inventory line, revid) for the fulltexts of
            all of the xml inventories specified by revision_keys.
        """
    stream = self.inventories.get_record_stream(keys, 'unordered', True)
    for record in stream:
        if record.storage_kind != 'absent':
            revid = record.key[-1]
            for line in record.get_bytes_as('lines'):
                yield (line, revid)