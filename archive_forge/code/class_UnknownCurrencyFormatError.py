from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
class UnknownCurrencyFormatError(KeyError):
    """Exception raised when an unknown currency format is requested."""