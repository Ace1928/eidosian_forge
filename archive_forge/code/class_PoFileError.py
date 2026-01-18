from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
class PoFileError(Exception):
    """Exception thrown by PoParser when an invalid po file is encountered."""

    def __init__(self, message: str, catalog: Catalog, line: str, lineno: int) -> None:
        super().__init__(f'{message} on {lineno}')
        self.catalog = catalog
        self.line = line
        self.lineno = lineno