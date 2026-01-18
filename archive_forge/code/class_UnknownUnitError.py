from __future__ import annotations
import decimal
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.numbers import LC_NUMERIC, format_decimal
class UnknownUnitError(ValueError):

    def __init__(self, unit: str, locale: Locale) -> None:
        ValueError.__init__(self, f'{unit} is not a known unit in {locale}')