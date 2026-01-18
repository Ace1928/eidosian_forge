from pythran.analyses import Inlinable, Aliases
from pythran.passmanager import Transformation
import gast as ast
import copy
class Inliner(ast.NodeTransformer):
    """ Helper transform that performed inlined body transformation. """

    def __init__(self, match):
        """ match : {original_variable_name : Arguments use on call}. """
        self.match = match
        super(Inliner, self).__init__()

    def visit_Name(self, node):
        """ Transform name from match values if available. """
        return self.match.get(node.id, node)

    def visit_Return(self, node):
        """ Remove return keyword after inline. """
        return self.visit(node.value)