import enum
from typing import Optional, List, Union, Iterable, Tuple
class SubscriptedIdentifier(Identifier):
    """An identifier with subscripted access."""

    def __init__(self, string: str, subscript: Union[Range, Expression]):
        super().__init__(string)
        self.subscript = subscript