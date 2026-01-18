from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def _remove_trailing_zeros_after_decimal(string: str, decimal_symbol: str) -> str:
    """
    Remove trailing zeros from the decimal part of a numeric string.

    This function takes a string representing a numeric value and a decimal symbol.
    It removes any trailing zeros that appear after the decimal symbol in the number.
    If the decimal part becomes empty after removing trailing zeros, the decimal symbol
    is also removed. If the string does not contain the decimal symbol, it is returned unchanged.

    :param string: The numeric string from which to remove trailing zeros.
    :type string: str
    :param decimal_symbol: The symbol used to denote the decimal point.
    :type decimal_symbol: str
    :return: The numeric string with trailing zeros removed from its decimal part.
    :rtype: str

    Example:
    >>> _remove_trailing_zeros_after_decimal("123.4500", ".")
    '123.45'
    >>> _remove_trailing_zeros_after_decimal("100.000", ".")
    '100'
    >>> _remove_trailing_zeros_after_decimal("100", ".")
    '100'
    """
    integer_part, _, decimal_part = string.partition(decimal_symbol)
    if decimal_part:
        decimal_part = decimal_part.rstrip('0')
        if decimal_part:
            return integer_part + decimal_symbol + decimal_part
        return integer_part
    return string