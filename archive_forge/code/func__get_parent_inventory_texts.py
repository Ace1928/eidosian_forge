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
def _get_parent_inventory_texts(self, inventory_text_cache, inventory_cache, parent_ids):
    cached_parent_texts = {}
    remaining_parent_ids = []
    for parent_id in parent_ids:
        p_text = inventory_text_cache.get(parent_id, None)
        if p_text is None:
            remaining_parent_ids.append(parent_id)
        else:
            cached_parent_texts[parent_id] = p_text
    ghosts = ()
    if remaining_parent_ids:
        parent_keys = [(r,) for r in remaining_parent_ids]
        present_parent_map = self._repository.inventories.get_parent_map(parent_keys)
        present_parent_ids = []
        ghosts = set()
        for p_id in remaining_parent_ids:
            if (p_id,) in present_parent_map:
                present_parent_ids.append(p_id)
            else:
                ghosts.add(p_id)
        to_lines = self._source_serializer.write_inventory_to_chunks
        for parent_inv in self._repository.iter_inventories(present_parent_ids):
            p_text = b''.join(to_lines(parent_inv))
            inventory_cache[parent_inv.revision_id] = parent_inv
            cached_parent_texts[parent_inv.revision_id] = p_text
            inventory_text_cache[parent_inv.revision_id] = p_text
    parent_texts = [cached_parent_texts[parent_id] for parent_id in parent_ids if parent_id not in ghosts]
    return parent_texts