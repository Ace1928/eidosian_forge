import datetime
import functools
import logging
import re
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units
class AutoDateLocator(DateLocator):
    """
    On autoscale, this class picks the best `DateLocator` to set the view
    limits and the tick locations.

    Attributes
    ----------
    intervald : dict

        Mapping of tick frequencies to multiples allowed for that ticking.
        The default is ::

            self.intervald = {
                YEARLY  : [1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500,
                           1000, 2000, 4000, 5000, 10000],
                MONTHLY : [1, 2, 3, 4, 6],
                DAILY   : [1, 2, 3, 7, 14, 21],
                HOURLY  : [1, 2, 3, 4, 6, 12],
                MINUTELY: [1, 5, 10, 15, 30],
                SECONDLY: [1, 5, 10, 15, 30],
                MICROSECONDLY: [1, 2, 5, 10, 20, 50, 100, 200, 500,
                                1000, 2000, 5000, 10000, 20000, 50000,
                                100000, 200000, 500000, 1000000],
            }

        where the keys are defined in `dateutil.rrule`.

        The interval is used to specify multiples that are appropriate for
        the frequency of ticking. For instance, every 7 days is sensible
        for daily ticks, but for minutes/seconds, 15 or 30 make sense.

        When customizing, you should only modify the values for the existing
        keys. You should not add or delete entries.

        Example for forcing ticks every 3 hours::

            locator = AutoDateLocator()
            locator.intervald[HOURLY] = [3]  # only show every 3 hours
    """

    def __init__(self, tz=None, minticks=5, maxticks=None, interval_multiples=True):
        """
        Parameters
        ----------
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        minticks : int
            The minimum number of ticks desired; controls whether ticks occur
            yearly, monthly, etc.
        maxticks : int
            The maximum number of ticks desired; controls the interval between
            ticks (ticking every other, every 3, etc.).  For fine-grained
            control, this can be a dictionary mapping individual rrule
            frequency constants (YEARLY, MONTHLY, etc.) to their own maximum
            number of ticks.  This can be used to keep the number of ticks
            appropriate to the format chosen in `AutoDateFormatter`. Any
            frequency not specified in this dictionary is given a default
            value.
        interval_multiples : bool, default: True
            Whether ticks should be chosen to be multiple of the interval,
            locking them to 'nicer' locations.  For example, this will force
            the ticks to be at hours 0, 6, 12, 18 when hourly ticking is done
            at 6 hour intervals.
        """
        super().__init__(tz=tz)
        self._freq = YEARLY
        self._freqs = [YEARLY, MONTHLY, DAILY, HOURLY, MINUTELY, SECONDLY, MICROSECONDLY]
        self.minticks = minticks
        self.maxticks = {YEARLY: 11, MONTHLY: 12, DAILY: 11, HOURLY: 12, MINUTELY: 11, SECONDLY: 11, MICROSECONDLY: 8}
        if maxticks is not None:
            try:
                self.maxticks.update(maxticks)
            except TypeError:
                self.maxticks = dict.fromkeys(self._freqs, maxticks)
        self.interval_multiples = interval_multiples
        self.intervald = {YEARLY: [1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500, 1000, 2000, 4000, 5000, 10000], MONTHLY: [1, 2, 3, 4, 6], DAILY: [1, 2, 3, 7, 14, 21], HOURLY: [1, 2, 3, 4, 6, 12], MINUTELY: [1, 5, 10, 15, 30], SECONDLY: [1, 5, 10, 15, 30], MICROSECONDLY: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]}
        if interval_multiples:
            self.intervald[DAILY] = [1, 2, 4, 7, 14]
        self._byranges = [None, range(1, 13), range(1, 32), range(0, 24), range(0, 60), range(0, 60), None]

    def __call__(self):
        dmin, dmax = self.viewlim_to_dt()
        locator = self.get_locator(dmin, dmax)
        return locator()

    def tick_values(self, vmin, vmax):
        return self.get_locator(vmin, vmax).tick_values(vmin, vmax)

    def nonsingular(self, vmin, vmax):
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (date2num(datetime.date(1970, 1, 1)), date2num(datetime.date(1970, 1, 2)))
        if vmax < vmin:
            vmin, vmax = (vmax, vmin)
        if vmin == vmax:
            vmin = vmin - DAYS_PER_YEAR * 2
            vmax = vmax + DAYS_PER_YEAR * 2
        return (vmin, vmax)

    def _get_unit(self):
        if self._freq in [MICROSECONDLY]:
            return 1.0 / MUSECONDS_PER_DAY
        else:
            return RRuleLocator.get_unit_generic(self._freq)

    def get_locator(self, dmin, dmax):
        """Pick the best locator based on a distance."""
        delta = relativedelta(dmax, dmin)
        tdelta = dmax - dmin
        if dmin > dmax:
            delta = -delta
            tdelta = -tdelta
        numYears = float(delta.years)
        numMonths = numYears * MONTHS_PER_YEAR + delta.months
        numDays = tdelta.days
        numHours = numDays * HOURS_PER_DAY + delta.hours
        numMinutes = numHours * MIN_PER_HOUR + delta.minutes
        numSeconds = np.floor(tdelta.total_seconds())
        numMicroseconds = np.floor(tdelta.total_seconds() * 1000000.0)
        nums = [numYears, numMonths, numDays, numHours, numMinutes, numSeconds, numMicroseconds]
        use_rrule_locator = [True] * 6 + [False]
        byranges = [None, 1, 1, 0, 0, 0, None]
        for i, (freq, num) in enumerate(zip(self._freqs, nums)):
            if num < self.minticks:
                byranges[i] = None
                continue
            for interval in self.intervald[freq]:
                if num <= interval * (self.maxticks[freq] - 1):
                    break
            else:
                if not (self.interval_multiples and freq == DAILY):
                    _api.warn_external(f"AutoDateLocator was unable to pick an appropriate interval for this date range. It may be necessary to add an interval value to the AutoDateLocator's intervald dictionary. Defaulting to {interval}.")
            self._freq = freq
            if self._byranges[i] and self.interval_multiples:
                byranges[i] = self._byranges[i][::interval]
                if i in (DAILY, WEEKLY):
                    if interval == 14:
                        byranges[i] = [1, 15]
                    elif interval == 7:
                        byranges[i] = [1, 8, 15, 22]
                interval = 1
            else:
                byranges[i] = self._byranges[i]
            break
        else:
            interval = 1
        if freq == YEARLY and self.interval_multiples:
            locator = YearLocator(interval, tz=self.tz)
        elif use_rrule_locator[i]:
            _, bymonth, bymonthday, byhour, byminute, bysecond, _ = byranges
            rrule = rrulewrapper(self._freq, interval=interval, dtstart=dmin, until=dmax, bymonth=bymonth, bymonthday=bymonthday, byhour=byhour, byminute=byminute, bysecond=bysecond)
            locator = RRuleLocator(rrule, tz=self.tz)
        else:
            locator = MicrosecondLocator(interval, tz=self.tz)
            if date2num(dmin) > 70 * 365 and interval < 1000:
                _api.warn_external(f'Plotting microsecond time intervals for dates far from the epoch (time origin: {get_epoch()}) is not well-supported. See matplotlib.dates.set_epoch to change the epoch.')
        locator.set_axis(self.axis)
        return locator