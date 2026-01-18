from __future__ import annotations
import ast
import inspect
import textwrap
from typing import Any
class DocstringVisitor(ast.NodeVisitor):

    def __init__(self) -> None:
        super().__init__()
        self.target: str | None = None
        self.attrs: dict[str, str] = {}
        self.previous_node_type: type[ast.AST] | None = None

    def visit(self, node: ast.AST) -> Any:
        node_result = super().visit(node)
        self.previous_node_type = type(node)
        return node_result

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if isinstance(node.target, ast.Name):
            self.target = node.target.id

    def visit_Expr(self, node: ast.Expr) -> Any:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and (self.previous_node_type is ast.AnnAssign):
            docstring = inspect.cleandoc(node.value.value)
            if self.target:
                self.attrs[self.target] = docstring
            self.target = None