from __future__ import annotations
class AbstractChemenvError(Exception):
    """Abstract class for Chemenv errors."""

    def __init__(self, cls, method, msg):
        """
        Args:
            cls:
            method:
            msg:
        """
        self.cls = cls
        self.method = method
        self.msg = msg

    def __str__(self):
        return f'{self.cls}: {self.method}\n{self.msg!r}'