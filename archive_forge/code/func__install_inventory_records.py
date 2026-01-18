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
def _install_inventory_records(self, records):
    if self._info[b'serializer'] == self._repository._serializer.format_num and self._repository._serializer.support_altered_by_hack:
        return self._install_mp_records_keys(self._repository.inventories, records)
    inventory_text_cache = lru_cache.LRUSizeCache(10 * 1024 * 1024)
    inventory_cache = lru_cache.LRUCache(10)
    with ui.ui_factory.nested_progress_bar() as pb:
        num_records = len(records)
        for idx, (key, metadata, bytes) in enumerate(records):
            pb.update('installing inventory', idx, num_records)
            revision_id = key[-1]
            parent_ids = metadata[b'parents']
            p_texts = self._get_parent_inventory_texts(inventory_text_cache, inventory_cache, parent_ids)
            target_lines = multiparent.MultiParent.from_patch(bytes).to_lines(p_texts)
            sha1 = osutils.sha_strings(target_lines)
            if sha1 != metadata[b'sha1']:
                raise errors.BadBundle("Can't convert to target format")
            inventory_text_cache[revision_id] = b''.join(target_lines)
            target_inv = self._source_serializer.read_inventory_from_lines(target_lines)
            del target_lines
            self._handle_root(target_inv, parent_ids)
            parent_inv = None
            if parent_ids:
                parent_inv = inventory_cache.get(parent_ids[0], None)
            try:
                if parent_inv is None:
                    self._repository.add_inventory(revision_id, target_inv, parent_ids)
                else:
                    delta = target_inv._make_delta(parent_inv)
                    self._repository.add_inventory_by_delta(parent_ids[0], delta, revision_id, parent_ids)
            except serializer.UnsupportedInventoryKind:
                raise errors.IncompatibleRevision(repr(self._repository))
            inventory_cache[revision_id] = target_inv