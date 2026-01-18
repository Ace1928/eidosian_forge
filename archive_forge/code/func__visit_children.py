import ast
def _visit_children(self, node):
    for child in ast.iter_child_nodes(node):
        super().visit(child)