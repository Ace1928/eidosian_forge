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
class schedule(BaseSchedule):
    """Schedule for periodic task.

    Arguments:
        run_every (float, ~datetime.timedelta): Time interval.
        relative (bool):  If set to True the run time will be rounded to the
            resolution of the interval.
        nowfun (Callable): Function returning the current date and time
            (:class:`~datetime.datetime`).
        app (Celery): Celery app instance.
    """
    relative: bool = False

    def __init__(self, run_every: float | timedelta | None=None, relative: bool=False, nowfun: Callable | None=None, app: Celery | None=None) -> None:
        self.run_every = maybe_timedelta(run_every)
        self.relative = relative
        super().__init__(nowfun=nowfun, app=app)

    def remaining_estimate(self, last_run_at: datetime) -> timedelta:
        return remaining(self.maybe_make_aware(last_run_at), self.run_every, self.maybe_make_aware(self.now()), self.relative)

    def is_due(self, last_run_at: datetime) -> tuple[bool, datetime]:
        """Return tuple of ``(is_due, next_time_to_check)``.

        Notes:
            - next time to check is in seconds.

            - ``(True, 20)``, means the task should be run now, and the next
                time to check is in 20 seconds.

            - ``(False, 12.3)``, means the task is not due, but that the
              scheduler should check again in 12.3 seconds.

        The next time to check is used to save energy/CPU cycles,
        it does not need to be accurate but will influence the precision
        of your schedule.  You must also keep in mind
        the value of :setting:`beat_max_loop_interval`,
        that decides the maximum number of seconds the scheduler can
        sleep between re-checking the periodic task intervals.  So if you
        have a task that changes schedule at run-time then your next_run_at
        check will decide how long it will take before a change to the
        schedule takes effect.  The max loop interval takes precedence
        over the next check at value returned.

        .. admonition:: Scheduler max interval variance

            The default max loop interval may vary for different schedulers.
            For the default scheduler the value is 5 minutes, but for example
            the :pypi:`django-celery-beat` database scheduler the value
            is 5 seconds.
        """
        last_run_at = self.maybe_make_aware(last_run_at)
        rem_delta = self.remaining_estimate(last_run_at)
        remaining_s = max(rem_delta.total_seconds(), 0)
        if remaining_s == 0:
            return schedstate(is_due=True, next=self.seconds)
        return schedstate(is_due=False, next=remaining_s)

    def __repr__(self) -> str:
        return f'<freq: {self.human_seconds}>'

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, schedule):
            return self.run_every == other.run_every
        return self.run_every == other

    def __reduce__(self) -> tuple[type, tuple[timedelta, bool, Callable | None]]:
        return (self.__class__, (self.run_every, self.relative, self.nowfun))

    @property
    def seconds(self) -> int | float:
        return max(self.run_every.total_seconds(), 0)

    @property
    def human_seconds(self) -> str:
        return humanize_seconds(self.seconds)