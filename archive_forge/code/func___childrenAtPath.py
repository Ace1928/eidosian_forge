from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def __childrenAtPath(self, parts):
    result = []
    node = self
    ancestors = parts[:-1]
    leaf = parts[-1]
    for name in ancestors:
        ns = None
        prefix, name = splitPrefix(name)
        if prefix is not None:
            ns = node.resolvePrefix(prefix)
        child = node.getChild(name, ns)
        if child is None:
            break
        node = child
    if child is not None:
        ns = None
        prefix, leaf = splitPrefix(leaf)
        if prefix is not None:
            ns = node.resolvePrefix(prefix)
        result = child.getChildren(leaf)
    return result