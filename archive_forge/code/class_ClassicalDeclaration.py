import enum
from typing import Optional, List, Union, Iterable, Tuple
class ClassicalDeclaration(Statement):
    """Declaration of a classical type, optionally initialising it to a value."""

    def __init__(self, type_: ClassicalType, identifier: Identifier, initializer=None):
        self.type = type_
        self.identifier = identifier
        self.initializer = initializer