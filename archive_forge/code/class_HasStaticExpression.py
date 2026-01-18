from pythran.passmanager import NodeAnalysis
class HasStaticExpression(NodeAnalysis):

    def __init__(self):
        self.result = False
        super(HasStaticExpression, self).__init__()

    def visit_Attribute(self, node):
        self.generic_visit(node)
        self.result |= node.attr == 'is_none'