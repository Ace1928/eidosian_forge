from __future__ import annotations
import functools
import warnings
from textwrap import dedent
from typing import Optional, Type, Union
class PlotnineError(Exception):
    """
    Exception for ggplot errors
    """

    def __init__(self, *args: str):
        args = tuple((dedent(arg) for arg in args))
        self.message = ' '.join(args)

    def __str__(self) -> str:
        """
        Error Message
        """
        return repr(self.message)