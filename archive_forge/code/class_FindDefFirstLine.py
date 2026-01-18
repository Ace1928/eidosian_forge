import ast
class FindDefFirstLine(ast.NodeVisitor):
    """
    Attributes
    ----------
    first_stmt_line : int or None
        This stores the first statement line number if the definition is found.
        Or, ``None`` if the definition is not found.
    """

    def __init__(self, code):
        """
        Parameters
        ----------
        code :
            The function's code object.
        """
        self._co_name = code.co_name
        self._co_firstlineno = code.co_firstlineno
        self.first_stmt_line = None

    def _visit_children(self, node):
        for child in ast.iter_child_nodes(node):
            super().visit(child)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == self._co_name:
            possible_start_lines = set([node.lineno])
            if node.decorator_list:
                first_decor = node.decorator_list[0]
                possible_start_lines.add(first_decor.lineno)
            if self._co_firstlineno in possible_start_lines:
                if node.body:
                    first_stmt = node.body[0]
                    if _is_docstring(first_stmt):
                        first_stmt = node.body[1]
                    self.first_stmt_line = first_stmt.lineno
                    return
                else:
                    pass
        self._visit_children(node)