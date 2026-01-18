from pythran.passmanager import FunctionAnalysis
import gast as ast
class Literals(FunctionAnalysis):
    """
        Store variable that save only Literals (with no construction cost)
    """

    def __init__(self):
        self.result = set()
        super(Literals, self).__init__()

    def visit_Assign(self, node):
        if isinstance(node.value, (ast.Constant, ast.Lambda)):
            targets = [target for target in node.targets if isinstance(target, ast.Name)]
            self.result.update(targets)