from __future__ import annotations
import locale
import sys
import typing as t
class LocaleError(SystemExit):
    """Exception to raise when locale related errors occur."""

    def __init__(self, message: str) -> None:
        super().__init__(f'ERROR: {message}')