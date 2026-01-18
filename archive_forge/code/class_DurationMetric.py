from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
class DurationMetric(NumericValuesContainer):
    """
    A container for duration metrics
    """
    name: Optional[str] = 'duration'

    @classmethod
    def dformat_duration(cls, duration: float, pretty: bool=True, short: int=0, include_ms: bool=False, as_int: bool=False) -> Dict[str, Union[float, int]]:
        """
        Formats a duration (secs) into a dict
        """
        if not pretty:
            unit = 'secs' if short else 'seconds'
            value = int(duration) if as_int else duration
            return {unit: value}
        data = {}
        if duration >= 86400:
            unit = 'd' if short > 1 else 'day'
            days = duration // 86400
            if short < 2 and days > 1:
                unit += 's'
            duration -= days * 86400
            data[unit] = int(days) if as_int else days
        if duration >= 3600:
            unit = 'hr' if short else 'hour'
            if short > 1:
                unit = unit[0]
            hours = duration // 3600
            if short < 2 and hours > 1:
                unit += 's'
            duration -= hours * 3600
            data[unit] = int(hours) if as_int else hours
        if duration >= 60:
            unit = 'min' if short else 'minute'
            if short > 1:
                unit = unit[0]
            minutes = duration // 60
            if short < 2 and minutes > 1:
                unit += 's'
            duration -= minutes * 60
            data[unit] = int(minutes) if as_int else minutes
        if duration >= 1:
            unit = 'sec' if short else 'second'
            if short > 1:
                unit = unit[0]
            if short < 2 and duration > 1:
                unit += 's'
            if include_ms:
                seconds = int(duration)
                duration -= seconds
                data[unit] = seconds
            elif short > 1:
                data[unit] = int(duration) if as_int else duration
            else:
                data[unit] = float(f'{duration:.2f}')
        if include_ms and duration > 0:
            unit = 'ms' if short else 'millisecond'
            milliseconds = int(duration * 1000)
            data[unit] = milliseconds
        return data

    @classmethod
    def pformat_duration(cls, duration: float, pretty: bool=True, short: int=0, include_ms: bool=False) -> str:
        """
        Formats a duration (secs) into a string

        535003.0 -> 5 days, 5 hours, 50 minutes, 3 seconds
        3593.0 -> 59 minutes, 53 seconds
        """
        data = cls.dformat_duration(duration=duration, pretty=pretty, short=short, include_ms=include_ms, as_int=True)
        if not data:
            return '0 secs'
        sep = '' if short > 1 else ' '
        if short > 2:
            return ''.join([f'{v}{sep}{k}' for k, v in data.items()])
        return ' '.join([f'{v}{sep}{k}' for k, v in data.items()]) if short else ', '.join([f'{v}{sep}{k}' for k, v in data.items()])

    def pretty(self, short: int=0, include_ms: bool=False) -> str:
        """
        Returns the pretty representation of the values
        """
        return self.pformat_duration(self.total, short=short, include_ms=include_ms)

    def __repr__(self) -> str:
        """
        Returns the representation of the values
        """
        name = self.name or self.__class__.__name__
        return f'<{name}>(total: {self.pformat_duration(self.total)}, average: {self.average:.2f}/sec)'

    def __str__(self) -> str:
        """
        Returns the string representation of the values
        """
        return f'{self.pformat_duration(self.total)}'

    @property
    def total_s(self) -> str:
        """
        Returns the total in seconds
        """
        return self.pformat_duration(self.total, short=1, include_ms=False)

    @property
    def average_s(self) -> str:
        """
        Returns the average in seconds
        """
        return self.pformat_duration(self.average, short=1, include_ms=False)

    @property
    def median_s(self) -> str:
        """
        Returns the median in seconds
        """
        return self.pformat_duration(self.median, short=1, include_ms=False)