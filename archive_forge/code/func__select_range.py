from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def _select_range(self, multiselect, keep_anchor, node, idx):
    """Selects a range between self._anchor and node or idx.
        If multiselect is True, it will be added to the selection, otherwise
        it will unselect everything before selecting the range. This is only
        called if self.multiselect is True.
        If keep anchor is False, the anchor is moved to node. This should
        always be True for keyboard selection.
        """
    select = self.select_node
    sister_nodes = self.get_selectable_nodes()
    end = len(sister_nodes) - 1
    last_node = self._anchor
    last_idx = self._anchor_idx
    if last_node is None:
        last_idx = end
        last_node = sister_nodes[end]
    elif last_idx > end or sister_nodes[last_idx] != last_node:
        try:
            last_idx = self.get_index_of_node(last_node, sister_nodes)
        except ValueError:
            return
    if idx > end or sister_nodes[idx] != node:
        try:
            idx = self.get_index_of_node(node, sister_nodes)
        except ValueError:
            return
    if last_idx > idx:
        last_idx, idx = (idx, last_idx)
    if not multiselect:
        self.clear_selection()
    for item in sister_nodes[last_idx:idx + 1]:
        select(item)
    if keep_anchor:
        self._anchor = last_node
        self._anchor_idx = last_idx
    else:
        self._anchor = node
        self._anchor_idx = idx
    self._last_selected_node = node
    self._last_node_idx = idx