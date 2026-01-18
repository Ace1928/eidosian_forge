from collections import deque
from typing import Deque
class TypeCheckWarning(UserWarning):
    """Emitted by typeguard's type checkers when a type mismatch is detected."""

    def __init__(self, message: str):
        super().__init__(message)