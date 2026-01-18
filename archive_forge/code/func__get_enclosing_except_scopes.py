import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def _get_enclosing_except_scopes(self, stop_at):
    included = []
    for node in reversed(self.lexical_scopes):
        if isinstance(node, gast.Try) and node.handlers:
            included.extend(node.handlers)
        if isinstance(node, stop_at):
            break
    return included