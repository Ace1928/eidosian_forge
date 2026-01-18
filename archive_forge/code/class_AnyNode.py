from __future__ import annotations
import re
class AnyNode(Node):
    """
    Union operation (OR operation) between several grammars. You don't
    initialize this yourself, but it's a result of a "Grammar1 | Grammar2"
    operation.
    """

    def __init__(self, children: list[Node]) -> None:
        self.children = children

    def __or__(self, other_node: Node) -> AnyNode:
        return AnyNode(self.children + [other_node])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.children!r})'