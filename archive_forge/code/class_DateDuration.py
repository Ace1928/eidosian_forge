from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from isoduration.formatter import format_duration
from isoduration.operations import add
@dataclass
class DateDuration:
    years: Decimal = Decimal(0)
    months: Decimal = Decimal(0)
    days: Decimal = Decimal(0)
    weeks: Decimal = Decimal(0)

    def __neg__(self) -> DateDuration:
        return DateDuration(years=-self.years, months=-self.months, days=-self.days, weeks=-self.weeks)