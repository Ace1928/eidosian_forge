from __future__ import annotations
from .visitor import AstVisitor
import typing as T
class AstIDGenerator(AstVisitor):

    def __init__(self) -> None:
        self.counter: T.Dict[str, int] = {}

    def visit_default_func(self, node: mparser.BaseNode) -> None:
        name = type(node).__name__
        if name not in self.counter:
            self.counter[name] = 0
        node.ast_id = name + '#' + str(self.counter[name])
        self.counter[name] += 1