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
class solar(BaseSchedule):
    """Solar event.

    A solar event can be used as the ``run_every`` value of a
    periodic task entry to schedule based on certain solar events.

    Notes:

        Available event values are:

            - ``dawn_astronomical``
            - ``dawn_nautical``
            - ``dawn_civil``
            - ``sunrise``
            - ``solar_noon``
            - ``sunset``
            - ``dusk_civil``
            - ``dusk_nautical``
            - ``dusk_astronomical``

    Arguments:
        event (str): Solar event that triggers this task.
            See note for available values.
        lat (float): The latitude of the observer.
        lon (float): The longitude of the observer.
        nowfun (Callable): Function returning the current date and time
            as a class:`~datetime.datetime`.
        app (Celery): Celery app instance.
    """
    _all_events = {'dawn_astronomical', 'dawn_nautical', 'dawn_civil', 'sunrise', 'solar_noon', 'sunset', 'dusk_civil', 'dusk_nautical', 'dusk_astronomical'}
    _horizons = {'dawn_astronomical': '-18', 'dawn_nautical': '-12', 'dawn_civil': '-6', 'sunrise': '-0:34', 'solar_noon': '0', 'sunset': '-0:34', 'dusk_civil': '-6', 'dusk_nautical': '-12', 'dusk_astronomical': '18'}
    _methods = {'dawn_astronomical': 'next_rising', 'dawn_nautical': 'next_rising', 'dawn_civil': 'next_rising', 'sunrise': 'next_rising', 'solar_noon': 'next_transit', 'sunset': 'next_setting', 'dusk_civil': 'next_setting', 'dusk_nautical': 'next_setting', 'dusk_astronomical': 'next_setting'}
    _use_center_l = {'dawn_astronomical': True, 'dawn_nautical': True, 'dawn_civil': True, 'sunrise': False, 'solar_noon': False, 'sunset': False, 'dusk_civil': True, 'dusk_nautical': True, 'dusk_astronomical': True}

    def __init__(self, event: str, lat: int | float, lon: int | float, **kwargs: Any) -> None:
        self.ephem = __import__('ephem')
        self.event = event
        self.lat = lat
        self.lon = lon
        super().__init__(**kwargs)
        if event not in self._all_events:
            raise ValueError(SOLAR_INVALID_EVENT.format(event=event, all_events=', '.join(sorted(self._all_events))))
        if lat < -90 or lat > 90:
            raise ValueError(SOLAR_INVALID_LATITUDE.format(lat=lat))
        if lon < -180 or lon > 180:
            raise ValueError(SOLAR_INVALID_LONGITUDE.format(lon=lon))
        cal = self.ephem.Observer()
        cal.lat = str(lat)
        cal.lon = str(lon)
        cal.elev = 0
        cal.horizon = self._horizons[event]
        cal.pressure = 0
        self.cal = cal
        self.method = self._methods[event]
        self.use_center = self._use_center_l[event]

    def __reduce__(self) -> tuple[type, tuple[str, int | float, int | float]]:
        return (self.__class__, (self.event, self.lat, self.lon))

    def __repr__(self) -> str:
        return '<solar: {} at latitude {}, longitude: {}>'.format(self.event, self.lat, self.lon)

    def remaining_estimate(self, last_run_at: datetime) -> timedelta:
        """Return estimate of next time to run.

        Returns:
            ~datetime.timedelta: when the periodic task should
                run next, or if it shouldn't run today (e.g., the sun does
                not rise today), returns the time when the next check
                should take place.
        """
        last_run_at = self.maybe_make_aware(last_run_at)
        last_run_at_utc = localize(last_run_at, timezone.utc)
        self.cal.date = last_run_at_utc
        try:
            if self.use_center:
                next_utc = getattr(self.cal, self.method)(self.ephem.Sun(), start=last_run_at_utc, use_center=self.use_center)
            else:
                next_utc = getattr(self.cal, self.method)(self.ephem.Sun(), start=last_run_at_utc)
        except self.ephem.CircumpolarError:
            next_utc = self.cal.next_antitransit(self.ephem.Sun()) + timedelta(minutes=1)
        next = self.maybe_make_aware(next_utc.datetime())
        now = self.maybe_make_aware(self.now())
        delta = next - now
        return delta

    def is_due(self, last_run_at: datetime) -> tuple[bool, datetime]:
        """Return tuple of ``(is_due, next_time_to_run)``.

        Note:
            next time to run is in seconds.

        See Also:
            :meth:`celery.schedules.schedule.is_due` for more information.
        """
        rem_delta = self.remaining_estimate(last_run_at)
        rem = max(rem_delta.total_seconds(), 0)
        due = rem == 0
        if due:
            rem_delta = self.remaining_estimate(self.now())
            rem = max(rem_delta.total_seconds(), 0)
        return schedstate(due, rem)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, solar):
            return other.event == self.event and other.lat == self.lat and (other.lon == self.lon)
        return NotImplemented