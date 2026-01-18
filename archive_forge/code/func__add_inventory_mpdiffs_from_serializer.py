import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
def _add_inventory_mpdiffs_from_serializer(self, revision_order):
    """Generate mpdiffs by serializing inventories.

        The current repository only has part of the tree shape information in
        the 'inventories' vf. So we use serializer.write_inventory_to_lines to
        get a 'full' representation of the tree shape, and then generate
        mpdiffs on that data stream. This stream can then be reconstructed on
        the other side.
        """
    inventory_key_order = [(r,) for r in revision_order]
    generator = _MPDiffInventoryGenerator(self.repository, inventory_key_order)
    for revision_id, parent_ids, sha1, diff in generator.iter_diffs():
        text = b''.join(diff.to_patch())
        self.bundle.add_multiparent_record(text, sha1, parent_ids, 'inventory', revision_id, None)