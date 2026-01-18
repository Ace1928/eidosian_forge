from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from ..util.deprecation import deprecated
from ..util.strings import format_docstring
from ..util.warnings import warn
from .tickers import Ticker
def RELATIVE_DATETIME_CONTEXT() -> DatetimeTickFormatter:
    return DatetimeTickFormatter(microseconds='%T', milliseconds='%T', seconds='%H:%M', minsec='%Hh', minutes='%Hh', hourmin='%F', hours='%F', days='%Y', months='', years='')