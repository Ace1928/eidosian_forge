from collections import deque
from . import errors, revision
def _initialize_nodes(self, parent_map):
    """Populate self._nodes.

        After this has finished:
        - self._nodes will have an entry for every entry in parent_map.
        - ghosts will have a parent_keys = None,
        - all nodes found will also have .child_keys populated with all known
          child_keys,
        """
    nodes = self._nodes
    for key, parent_keys in parent_map.items():
        if key in nodes:
            node = nodes[key]
            node.parent_keys = parent_keys
        else:
            node = _KnownGraphNode(key, parent_keys)
            nodes[key] = node
        for parent_key in parent_keys:
            try:
                parent_node = nodes[parent_key]
            except KeyError:
                parent_node = _KnownGraphNode(parent_key, None)
                nodes[parent_key] = parent_node
            parent_node.child_keys.append(key)