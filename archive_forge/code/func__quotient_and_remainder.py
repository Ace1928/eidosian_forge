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
def _quotient_and_remainder(value: float, divisor: float, unit: Unit, minimum_unit: Unit, suppress: collections.abc.Iterable[Unit]) -> tuple[float, float]:
    """Divide `value` by `divisor` returning the quotient and remainder.

    If `unit` is `minimum_unit`, makes the quotient a float number and the remainder
    will be zero. The rational is that if `unit` is the unit of the quotient, we cannot
    represent the remainder because it would require a unit smaller than the
    `minimum_unit`.

    >>> from humanize.time import _quotient_and_remainder, Unit
    >>> _quotient_and_remainder(36, 24, Unit.DAYS, Unit.DAYS, [])
    (1.5, 0)

    If unit is in `suppress`, the quotient will be zero and the remainder will be the
    initial value. The idea is that if we cannot use `unit`, we are forced to use a
    lower unit so we cannot do the division.

    >>> _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [Unit.DAYS])
    (0, 36)

    In other case return quotient and remainder as `divmod` would do it.

    >>> _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [])
    (1, 12)

    """
    if unit == minimum_unit:
        return (value / divisor, 0)
    if unit in suppress:
        return (0, value)
    return divmod(value, divisor)