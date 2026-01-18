import enum
from typing import Optional, List, Union, Iterable, Tuple
class ForLoopStatement(Statement):
    """
    AST node for ``for`` loops.

    ::

        ForLoop: "for" Identifier "in" SetDeclaration ProgramBlock
        SetDeclaration:
            | Identifier
            | "{" Expression ("," Expression)* "}"
            | "[" Range "]"
    """

    def __init__(self, indexset: Union[Identifier, IndexSet, Range], parameter: Identifier, body: ProgramBlock):
        self.indexset = indexset
        self.parameter = parameter
        self.body = body