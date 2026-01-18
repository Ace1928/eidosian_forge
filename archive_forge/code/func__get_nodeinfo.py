from types import GenericAlias
def _get_nodeinfo(self, node):
    if (result := self._node2info.get(node)) is None:
        self._node2info[node] = result = _NodeInfo(node)
    return result