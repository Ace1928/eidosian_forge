from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def _resolve_last_node(self):
    sister_nodes = self.get_selectable_nodes()
    if not len(sister_nodes):
        return (None, 0)
    last_node = self._last_selected_node
    last_idx = self._last_node_idx
    end = len(sister_nodes) - 1
    if last_node is None:
        last_node = self._anchor
        last_idx = self._anchor_idx
    if last_node is None:
        return (sister_nodes[end], end)
    if last_idx > end or sister_nodes[last_idx] != last_node:
        try:
            return (last_node, self.get_index_of_node(last_node, sister_nodes))
        except ValueError:
            return (sister_nodes[end], end)
    return (last_node, last_idx)