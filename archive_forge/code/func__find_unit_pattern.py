from __future__ import annotations
import decimal
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.numbers import LC_NUMERIC, format_decimal
def _find_unit_pattern(unit_id: str, locale: Locale | str | None=LC_NUMERIC) -> str | None:
    """
    Expand a unit into a qualified form.

    Known units can be found in the CLDR Unit Validity XML file:
    https://unicode.org/repos/cldr/tags/latest/common/validity/unit.xml

    >>> _find_unit_pattern("radian", locale="en")
    'angle-radian'

    Unknown values will return None.

    >>> _find_unit_pattern("horse", locale="en")

    :param unit_id: the code of a measurement unit.
    :return: A key to the `unit_patterns` mapping, or None.
    """
    locale = Locale.parse(locale)
    unit_patterns = locale._data['unit_patterns']
    if unit_id in unit_patterns:
        return unit_id
    for unit_pattern in sorted(unit_patterns, key=len):
        if unit_pattern.endswith(unit_id):
            return unit_pattern
    return None