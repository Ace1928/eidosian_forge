from __future__ import annotations
from .charset import Charset
class VariableInvalidError(Exception):
    """Exception thrown for invalid variables."""
    variable: str

    def __init__(self, variable: str) -> None:
        self.variable = variable

    def __str__(self) -> str:
        """Convert to string."""
        return 'Bad variable: ' + self.variable