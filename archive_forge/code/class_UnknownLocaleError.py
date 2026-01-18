from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
class UnknownLocaleError(Exception):
    """Exception thrown when a locale is requested for which no locale data
    is available.
    """

    def __init__(self, identifier: str) -> None:
        """Create the exception.

        :param identifier: the identifier string of the unsupported locale
        """
        Exception.__init__(self, f'unknown locale {identifier!r}')
        self.identifier = identifier