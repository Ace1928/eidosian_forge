from __future__ import annotations
import re
from bisect import bisect, bisect_left
from collections import namedtuple
from collections.abc import Iterable
from datetime import datetime, timedelta, tzinfo
from typing import Any, Callable, Mapping, Sequence
from kombu.utils.objects import cached_property
from celery import Celery
from . import current_app
from .utils.collections import AttributeDict
from .utils.time import (ffwd, humanize_seconds, localize, maybe_make_aware, maybe_timedelta, remaining, timezone,
class crontab(BaseSchedule):
    """Crontab schedule.

    A Crontab can be used as the ``run_every`` value of a
    periodic task entry to add :manpage:`crontab(5)`-like scheduling.

    Like a :manpage:`cron(5)`-job, you can specify units of time of when
    you'd like the task to execute.  It's a reasonably complete
    implementation of :command:`cron`'s features, so it should provide a fair
    degree of scheduling needs.

    You can specify a minute, an hour, a day of the week, a day of the
    month, and/or a month in the year in any of the following formats:

    .. attribute:: minute

        - A (list of) integers from 0-59 that represent the minutes of
          an hour of when execution should occur; or
        - A string representing a Crontab pattern.  This may get pretty
          advanced, like ``minute='*/15'`` (for every quarter) or
          ``minute='1,13,30-45,50-59/2'``.

    .. attribute:: hour

        - A (list of) integers from 0-23 that represent the hours of
          a day of when execution should occur; or
        - A string representing a Crontab pattern.  This may get pretty
          advanced, like ``hour='*/3'`` (for every three hours) or
          ``hour='0,8-17/2'`` (at midnight, and every two hours during
          office hours).

    .. attribute:: day_of_week

        - A (list of) integers from 0-6, where Sunday = 0 and Saturday =
          6, that represent the days of a week that execution should
          occur.
        - A string representing a Crontab pattern.  This may get pretty
          advanced, like ``day_of_week='mon-fri'`` (for weekdays only).
          (Beware that ``day_of_week='*/2'`` does not literally mean
          'every two days', but 'every day that is divisible by two'!)

    .. attribute:: day_of_month

        - A (list of) integers from 1-31 that represents the days of the
          month that execution should occur.
        - A string representing a Crontab pattern.  This may get pretty
          advanced, such as ``day_of_month='2-30/2'`` (for every even
          numbered day) or ``day_of_month='1-7,15-21'`` (for the first and
          third weeks of the month).

    .. attribute:: month_of_year

        - A (list of) integers from 1-12 that represents the months of
          the year during which execution can occur.
        - A string representing a Crontab pattern.  This may get pretty
          advanced, such as ``month_of_year='*/3'`` (for the first month
          of every quarter) or ``month_of_year='2-12/2'`` (for every even
          numbered month).

    .. attribute:: nowfun

        Function returning the current date and time
        (:class:`~datetime.datetime`).

    .. attribute:: app

        The Celery app instance.

    It's important to realize that any day on which execution should
    occur must be represented by entries in all three of the day and
    month attributes.  For example, if ``day_of_week`` is 0 and
    ``day_of_month`` is every seventh day, only months that begin
    on Sunday and are also in the ``month_of_year`` attribute will have
    execution events.  Or, ``day_of_week`` is 1 and ``day_of_month``
    is '1-7,15-21' means every first and third Monday of every month
    present in ``month_of_year``.
    """

    def __init__(self, minute: str='*', hour: str='*', day_of_week: str='*', day_of_month: str='*', month_of_year: str='*', **kwargs: Any) -> None:
        self._orig_minute = cronfield(minute)
        self._orig_hour = cronfield(hour)
        self._orig_day_of_week = cronfield(day_of_week)
        self._orig_day_of_month = cronfield(day_of_month)
        self._orig_month_of_year = cronfield(month_of_year)
        self._orig_kwargs = kwargs
        self.hour = self._expand_cronspec(hour, 24)
        self.minute = self._expand_cronspec(minute, 60)
        self.day_of_week = self._expand_cronspec(day_of_week, 7)
        self.day_of_month = self._expand_cronspec(day_of_month, 31, 1)
        self.month_of_year = self._expand_cronspec(month_of_year, 12, 1)
        super().__init__(**kwargs)

    @staticmethod
    def _expand_cronspec(cronspec: int | str | Iterable, max_: int, min_: int=0) -> set[Any]:
        """Expand cron specification.

        Takes the given cronspec argument in one of the forms:

        .. code-block:: text

            int         (like 7)
            str         (like '3-5,*/15', '*', or 'monday')
            set         (like {0,15,30,45}
            list        (like [8-17])

        And convert it to an (expanded) set representing all time unit
        values on which the Crontab triggers.  Only in case of the base
        type being :class:`str`, parsing occurs.  (It's fast and
        happens only once for each Crontab instance, so there's no
        significant performance overhead involved.)

        For the other base types, merely Python type conversions happen.

        The argument ``max_`` is needed to determine the expansion of
        ``*`` and ranges.  The argument ``min_`` is needed to determine
        the expansion of ``*`` and ranges for 1-based cronspecs, such as
        day of month or month of year.  The default is sufficient for minute,
        hour, and day of week.
        """
        if isinstance(cronspec, int):
            result = {cronspec}
        elif isinstance(cronspec, str):
            result = crontab_parser(max_, min_).parse(cronspec)
        elif isinstance(cronspec, set):
            result = cronspec
        elif isinstance(cronspec, Iterable):
            result = set(cronspec)
        else:
            raise TypeError(CRON_INVALID_TYPE.format(type=type(cronspec)))
        for number in result:
            if number >= max_ + min_ or number < min_:
                raise ValueError(CRON_PATTERN_INVALID.format(min=min_, max=max_ - 1 + min_, value=number))
        return result

    def _delta_to_next(self, last_run_at: datetime, next_hour: int, next_minute: int) -> ffwd:
        """Find next delta.

        Takes a :class:`~datetime.datetime` of last run, next minute and hour,
        and returns a :class:`~celery.utils.time.ffwd` for the next
        scheduled day and time.

        Only called when ``day_of_month`` and/or ``month_of_year``
        cronspec is specified to further limit scheduled task execution.
        """
        datedata = AttributeDict(year=last_run_at.year)
        days_of_month = sorted(self.day_of_month)
        months_of_year = sorted(self.month_of_year)

        def day_out_of_range(year: int, month: int, day: int) -> bool:
            try:
                datetime(year=year, month=month, day=day)
            except ValueError:
                return True
            return False

        def is_before_last_run(year: int, month: int, day: int) -> bool:
            return self.maybe_make_aware(datetime(year, month, day, next_hour, next_minute), naive_as_utc=False) < last_run_at

        def roll_over() -> None:
            for _ in range(2000):
                flag = datedata.dom == len(days_of_month) or day_out_of_range(datedata.year, months_of_year[datedata.moy], days_of_month[datedata.dom]) or is_before_last_run(datedata.year, months_of_year[datedata.moy], days_of_month[datedata.dom])
                if flag:
                    datedata.dom = 0
                    datedata.moy += 1
                    if datedata.moy == len(months_of_year):
                        datedata.moy = 0
                        datedata.year += 1
                else:
                    break
            else:
                raise RuntimeError('unable to rollover, time specification is probably invalid')
        if last_run_at.month in self.month_of_year:
            datedata.dom = bisect(days_of_month, last_run_at.day)
            datedata.moy = bisect_left(months_of_year, last_run_at.month)
        else:
            datedata.dom = 0
            datedata.moy = bisect(months_of_year, last_run_at.month)
            if datedata.moy == len(months_of_year):
                datedata.moy = 0
        roll_over()
        while 1:
            th = datetime(year=datedata.year, month=months_of_year[datedata.moy], day=days_of_month[datedata.dom])
            if th.isoweekday() % 7 in self.day_of_week:
                break
            datedata.dom += 1
            roll_over()
        return ffwd(year=datedata.year, month=months_of_year[datedata.moy], day=days_of_month[datedata.dom], hour=next_hour, minute=next_minute, second=0, microsecond=0)

    def __repr__(self) -> str:
        return CRON_REPR.format(self)

    def __reduce__(self) -> tuple[type, tuple[str, str, str, str, str], Any]:
        return (self.__class__, (self._orig_minute, self._orig_hour, self._orig_day_of_week, self._orig_day_of_month, self._orig_month_of_year), self._orig_kwargs)

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        super().__init__(**state)

    def remaining_delta(self, last_run_at: datetime, tz: tzinfo | None=None, ffwd: type=ffwd) -> tuple[datetime, Any, datetime]:
        last_run_at = self.maybe_make_aware(last_run_at)
        now = self.maybe_make_aware(self.now())
        dow_num = last_run_at.isoweekday() % 7
        execute_this_date = last_run_at.month in self.month_of_year and last_run_at.day in self.day_of_month and (dow_num in self.day_of_week)
        execute_this_hour = execute_this_date and last_run_at.day == now.day and (last_run_at.month == now.month) and (last_run_at.year == now.year) and (last_run_at.hour in self.hour) and (last_run_at.minute < max(self.minute))
        if execute_this_hour:
            next_minute = min((minute for minute in self.minute if minute > last_run_at.minute))
            delta = ffwd(minute=next_minute, second=0, microsecond=0)
        else:
            next_minute = min(self.minute)
            execute_today = execute_this_date and last_run_at.hour < max(self.hour)
            if execute_today:
                next_hour = min((hour for hour in self.hour if hour > last_run_at.hour))
                delta = ffwd(hour=next_hour, minute=next_minute, second=0, microsecond=0)
            else:
                next_hour = min(self.hour)
                all_dom_moy = self._orig_day_of_month == '*' and self._orig_month_of_year == '*'
                if all_dom_moy:
                    next_day = min([day for day in self.day_of_week if day > dow_num] or self.day_of_week)
                    add_week = next_day == dow_num
                    delta = ffwd(weeks=add_week and 1 or 0, weekday=(next_day - 1) % 7, hour=next_hour, minute=next_minute, second=0, microsecond=0)
                else:
                    delta = self._delta_to_next(last_run_at, next_hour, next_minute)
        return (self.to_local(last_run_at), delta, self.to_local(now))

    def remaining_estimate(self, last_run_at: datetime, ffwd: type=ffwd) -> timedelta:
        """Estimate of next run time.

        Returns when the periodic task should run next as a
        :class:`~datetime.timedelta`.
        """
        return remaining(*self.remaining_delta(last_run_at, ffwd=ffwd))

    def is_due(self, last_run_at: datetime) -> tuple[bool, datetime]:
        """Return tuple of ``(is_due, next_time_to_run)``.

        If :setting:`beat_cron_starting_deadline`  has been specified, the
        scheduler will make sure that the `last_run_at` time is within the
        deadline. This prevents tasks that could have been run according to
        the crontab, but didn't, from running again unexpectedly.

        Note:
            Next time to run is in seconds.

        SeeAlso:
            :meth:`celery.schedules.schedule.is_due` for more information.
        """
        rem_delta = self.remaining_estimate(last_run_at)
        rem_secs = rem_delta.total_seconds()
        rem = max(rem_secs, 0)
        due = rem == 0
        deadline_secs = self.app.conf.beat_cron_starting_deadline
        has_passed_deadline = False
        if deadline_secs is not None:
            last_date_checked = last_run_at
            last_feasible_rem_secs = rem_secs
            while rem_secs < 0:
                last_date_checked = last_date_checked + abs(rem_delta)
                rem_delta = self.remaining_estimate(last_date_checked)
                rem_secs = rem_delta.total_seconds()
                if rem_secs < 0:
                    last_feasible_rem_secs = rem_secs
            has_passed_deadline = -last_feasible_rem_secs > deadline_secs
            if has_passed_deadline:
                due = False
        if due or has_passed_deadline:
            rem_delta = self.remaining_estimate(self.now())
            rem = max(rem_delta.total_seconds(), 0)
        return schedstate(due, rem)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, crontab):
            return other.month_of_year == self.month_of_year and other.day_of_month == self.day_of_month and (other.day_of_week == self.day_of_week) and (other.hour == self.hour) and (other.minute == self.minute) and super().__eq__(other)
        return NotImplemented