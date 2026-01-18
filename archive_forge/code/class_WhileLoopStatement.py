import enum
from typing import Optional, List, Union, Iterable, Tuple
class WhileLoopStatement(Statement):
    """
    AST node for ``while`` loops.

    ::

        WhileLoop: "while" "(" Expression ")" ProgramBlock
    """

    def __init__(self, condition: Expression, body: ProgramBlock):
        self.condition = condition
        self.body = body