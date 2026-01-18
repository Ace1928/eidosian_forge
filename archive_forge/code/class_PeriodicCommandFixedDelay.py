import datetime
import numbers
import abc
import bisect
import pytz
class PeriodicCommandFixedDelay(PeriodicCommand):
    """
    Like a periodic command, but don't calculate the delay based on
    the current time. Instead use a fixed delay following the initial
    run.
    """

    @classmethod
    def at_time(cls, at, delay, target):
        """
        >>> cmd = PeriodicCommandFixedDelay.at_time(0, 30, None)
        >>> cmd.delay.total_seconds()
        30.0
        """
        at = cls._from_timestamp(at)
        cmd = cls.from_datetime(at)
        if isinstance(delay, numbers.Number):
            delay = datetime.timedelta(seconds=delay)
        cmd.delay = delay
        cmd.target = target
        return cmd

    @classmethod
    def daily_at(cls, at, target):
        """
        Schedule a command to run at a specific time each day.

        >>> from tempora import utc
        >>> noon = utc.time(12, 0)
        >>> cmd = PeriodicCommandFixedDelay.daily_at(noon, None)
        >>> cmd.delay.total_seconds()
        86400.0
        """
        daily = datetime.timedelta(days=1)
        when = datetime.datetime.combine(datetime.date.today(), at)
        when -= daily
        while when < now():
            when += daily
        return cls.at_time(cls._localize(when), daily, target)