from __future__ import annotations
import collections.abc
import datetime as dt
import math
import typing
from enum import Enum
from functools import total_ordering
from typing import Any
from .i18n import _gettext as _
from .i18n import _ngettext
from .number import intcomma
def _suppress_lower_units(min_unit: Unit, suppress: typing.Iterable[Unit]) -> set[Unit]:
    """Extend suppressed units (if any) with all units lower than the minimum unit.

    >>> from humanize.time import _suppress_lower_units, Unit
    >>> [x.name for x in sorted(_suppress_lower_units(Unit.SECONDS, [Unit.DAYS]))]
    ['MICROSECONDS', 'MILLISECONDS', 'DAYS']
    """
    suppress = set(suppress)
    for unit in Unit:
        if unit == min_unit:
            break
        suppress.add(unit)
    return suppress