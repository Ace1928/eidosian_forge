from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def _init_chk_inventory(self, revision_id, root_id):
    """Generate a CHKInventory for a parentless revision."""
    from ...bzr import chk_map
    chk_store = self.repo.chk_bytes
    serializer = self.repo._format._serializer
    search_key_name = serializer.search_key_name
    maximum_size = serializer.maximum_size
    inv = inventory.CHKInventory(search_key_name)
    inv.revision_id = revision_id
    inv.root_id = root_id
    search_key_func = chk_map.search_key_registry.get(search_key_name)
    inv.id_to_entry = chk_map.CHKMap(chk_store, None, search_key_func)
    inv.id_to_entry._root_node.set_maximum_size(maximum_size)
    inv.parent_id_basename_to_file_id = chk_map.CHKMap(chk_store, None, search_key_func)
    inv.parent_id_basename_to_file_id._root_node.set_maximum_size(maximum_size)
    inv.parent_id_basename_to_file_id._root_node._key_width = 2
    return inv