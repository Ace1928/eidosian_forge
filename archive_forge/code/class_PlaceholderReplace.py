from pythran.conversion import mangle
from pythran.analyses import Check, Placeholder, AST_or
from pythran.passmanager import Transformation
from copy import deepcopy
import gast as ast
from inspect import isclass
class PlaceholderReplace(Transformation):
    """ Helper class to replace the placeholder once value is collected. """

    def __init__(self, placeholders):
        """ Store placeholders value collected. """
        self.placeholders = placeholders
        super(PlaceholderReplace, self).__init__()

    def visit(self, node):
        """ Replace the placeholder if it is one or continue. """
        if isinstance(node, Placeholder):
            return self.placeholders[node.id]
        else:
            return super(PlaceholderReplace, self).visit(node)