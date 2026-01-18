import re
import ast
from hacking import core
def _is_raised_later(self, node, name):

    def find_peers(node):
        node_for_line = node._parent
        for _field, value in ast.iter_fields(node._parent._parent):
            if isinstance(value, list) and node_for_line in value:
                return value[value.index(node_for_line) + 1:]
            continue
        return []
    peers = find_peers(node)
    for peer in peers:
        if isinstance(peer, ast.Raise):
            exc = peer.exc
            if isinstance(exc, ast.Call) and len(exc.args) > 0 and isinstance(exc.args[0], ast.Name) and (name in (a.id for a in exc.args)):
                return True
            else:
                return False
        elif isinstance(peer, ast.Assign):
            if name in (t.id for t in peer.targets if hasattr(t, 'id')):
                return False