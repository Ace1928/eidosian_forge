import enum
from typing import Optional, List, Union, Iterable, Tuple
class BitArrayType(ClassicalType):
    """Type information for a sized number of classical bits."""

    def __init__(self, size: int):
        self.size = size