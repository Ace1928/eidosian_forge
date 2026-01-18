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
def _do_generate_text_key_index(self, ancestors, text_key_references, pb):
    """Helper for _generate_text_key_index to avoid deep nesting."""
    revision_order = tsort.topo_sort(ancestors)
    invalid_keys = set()
    revision_keys = {}
    for revision_id in revision_order:
        revision_keys[revision_id] = set()
    text_count = len(text_key_references)
    text_key_cache = {}
    for text_key, valid in text_key_references.items():
        if not valid:
            invalid_keys.add(text_key)
        else:
            revision_keys[text_key[1]].add(text_key)
        text_key_cache[text_key] = text_key
    del text_key_references
    text_index = {}
    text_graph = graph.Graph(graph.DictParentsProvider(text_index))
    NULL_REVISION = _mod_revision.NULL_REVISION
    inventory_cache = lru_cache.LRUCache(10)
    batch_size = 10
    batch_count = len(revision_order) // batch_size + 1
    processed_texts = 0
    pb.update(gettext('Calculating text parents'), processed_texts, text_count)
    for offset in range(batch_count):
        to_query = revision_order[offset * batch_size:(offset + 1) * batch_size]
        if not to_query:
            break
        for revision_id in to_query:
            parent_ids = ancestors[revision_id]
            for text_key in revision_keys[revision_id]:
                pb.update(gettext('Calculating text parents'), processed_texts)
                processed_texts += 1
                candidate_parents = []
                for parent_id in parent_ids:
                    parent_text_key = (text_key[0], parent_id)
                    try:
                        check_parent = parent_text_key not in revision_keys[parent_id]
                    except KeyError:
                        check_parent = False
                        parent_text_key = None
                    if check_parent:
                        try:
                            inv = inventory_cache[parent_id]
                        except KeyError:
                            inv = self.revision_tree(parent_id).root_inventory
                            inventory_cache[parent_id] = inv
                        try:
                            parent_entry = inv.get_entry(text_key[0])
                        except (KeyError, errors.NoSuchId):
                            parent_entry = None
                        if parent_entry is not None:
                            parent_text_key = (text_key[0], parent_entry.revision)
                        else:
                            parent_text_key = None
                    if parent_text_key is not None:
                        candidate_parents.append(text_key_cache[parent_text_key])
                parent_heads = text_graph.heads(candidate_parents)
                new_parents = list(parent_heads)
                new_parents.sort(key=lambda x: candidate_parents.index(x))
                if new_parents == []:
                    new_parents = [NULL_REVISION]
                text_index[text_key] = new_parents
    for text_key in invalid_keys:
        text_index[text_key] = [NULL_REVISION]
    return text_index