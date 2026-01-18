import enum
from typing import Optional, List, Union, Iterable, Tuple
class AliasStatement(ASTNode):
    """
    aliasStatement
        : 'let' Identifier EQUALS indexIdentifier SEMICOLON
    """

    def __init__(self, identifier: Identifier, value: Expression):
        self.identifier = identifier
        self.value = value