from pythran.passmanager import ModuleAnalysis
class AncestorsWithBody(Ancestors):
    visit = ModuleAnalysis.visit

    def visit_metadata(self, node):
        if hasattr(node, 'metadata'):
            self.generic_visit(node.metadata)

    def visit_body(self, body):
        body_as_tuple = tuple(body)
        self.result[body_as_tuple] = current = self.current
        self.current += (body_as_tuple,)
        for stmt in body:
            self.generic_visit(stmt)
        self.current = current

    def visit_If(self, node):
        self.result[node] = current = self.current
        self.current += (node,)
        self.generic_visit(node.test)
        self.visit_metadata(node)
        self.visit_body(node.body)
        self.visit_body(node.orelse)
        self.current = current

    def visit_While(self, node):
        self.result[node] = current = self.current
        self.current += (node,)
        self.generic_visit(node.test)
        self.visit_metadata(node)
        self.visit_body(node.body)
        self.visit_body(node.orelse)
        self.current = current

    def visit_For(self, node):
        self.result[node] = current = self.current
        self.current += (node,)
        self.generic_visit(node.target)
        self.generic_visit(node.iter)
        self.visit_metadata(node)
        self.visit_body(node.body)
        self.visit_body(node.orelse)
        self.current = current

    def visit_Try(self, node):
        self.result[node] = current = self.current
        self.current += (node,)
        self.visit_metadata(node)
        self.visit_body(node.body)
        for handler in node.handlers:
            self.generic_visit(handler)
        self.visit_body(node.orelse)
        self.visit_body(node.finalbody)
        self.current = current