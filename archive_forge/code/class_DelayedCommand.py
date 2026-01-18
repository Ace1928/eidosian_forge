import datetime
import numbers
import abc
import bisect
import pytz
class DelayedCommand(datetime.datetime):
    """
    A command to be executed after some delay (seconds or timedelta).
    """

    @classmethod
    def from_datetime(cls, other):
        return cls(other.year, other.month, other.day, other.hour, other.minute, other.second, other.microsecond, other.tzinfo)

    @classmethod
    def after(cls, delay, target):
        if not isinstance(delay, datetime.timedelta):
            delay = datetime.timedelta(seconds=delay)
        due_time = now() + delay
        cmd = cls.from_datetime(due_time)
        cmd.delay = delay
        cmd.target = target
        return cmd

    @staticmethod
    def _from_timestamp(input):
        """
        If input is a real number, interpret it as a Unix timestamp
        (seconds sinc Epoch in UTC) and return a timezone-aware
        datetime object. Otherwise return input unchanged.
        """
        if not isinstance(input, numbers.Real):
            return input
        return from_timestamp(input)

    @classmethod
    def at_time(cls, at, target):
        """
        Construct a DelayedCommand to come due at `at`, where `at` may be
        a datetime or timestamp.
        """
        at = cls._from_timestamp(at)
        cmd = cls.from_datetime(at)
        cmd.delay = at - now()
        cmd.target = target
        return cmd

    def due(self):
        return now() >= self