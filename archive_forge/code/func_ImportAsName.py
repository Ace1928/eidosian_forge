from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def ImportAsName(name, as_name, prefix=None):
    new_name = Name(name)
    new_as = Name(u'as', prefix=u' ')
    new_as_name = Name(as_name, prefix=u' ')
    new_node = Node(syms.import_as_name, [new_name, new_as, new_as_name])
    if prefix is not None:
        new_node.prefix = prefix
    return new_node