import enum
from typing import Optional, List, Union, Iterable, Tuple
class ProgramBlock(ASTNode):
    """
    programBlock
        : statement | controlDirective
        | LBRACE(statement | controlDirective) * RBRACE
    """

    def __init__(self, statements: List[Statement]):
        self.statements = statements