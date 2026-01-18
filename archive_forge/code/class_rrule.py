import calendar
import datetime
import heapq
import itertools
import re
import sys
from functools import wraps
from warnings import warn
from six import advance_iterator, integer_types
from six.moves import _thread, range
from ._common import weekday as weekdaybase
class rrule(rrulebase):
    """
    That's the base of the rrule operation. It accepts all the keywords
    defined in the RFC as its constructor parameters (except byday,
    which was renamed to byweekday) and more. The constructor prototype is::

            rrule(freq)

    Where freq must be one of YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY,
    or SECONDLY.

    .. note::
        Per RFC section 3.3.10, recurrence instances falling on invalid dates
        and times are ignored rather than coerced:

            Recurrence rules may generate recurrence instances with an invalid
            date (e.g., February 30) or nonexistent local time (e.g., 1:30 AM
            on a day where the local time is moved forward by an hour at 1:00
            AM).  Such recurrence instances MUST be ignored and MUST NOT be
            counted as part of the recurrence set.

        This can lead to possibly surprising behavior when, for example, the
        start date occurs at the end of the month:

        >>> from dateutil.rrule import rrule, MONTHLY
        >>> from datetime import datetime
        >>> start_date = datetime(2014, 12, 31)
        >>> list(rrule(freq=MONTHLY, count=4, dtstart=start_date))
        ... # doctest: +NORMALIZE_WHITESPACE
        [datetime.datetime(2014, 12, 31, 0, 0),
         datetime.datetime(2015, 1, 31, 0, 0),
         datetime.datetime(2015, 3, 31, 0, 0),
         datetime.datetime(2015, 5, 31, 0, 0)]

    Additionally, it supports the following keyword arguments:

    :param dtstart:
        The recurrence start. Besides being the base for the recurrence,
        missing parameters in the final recurrence instances will also be
        extracted from this date. If not given, datetime.now() will be used
        instead.
    :param interval:
        The interval between each freq iteration. For example, when using
        YEARLY, an interval of 2 means once every two years, but with HOURLY,
        it means once every two hours. The default interval is 1.
    :param wkst:
        The week start day. Must be one of the MO, TU, WE constants, or an
        integer, specifying the first day of the week. This will affect
        recurrences based on weekly periods. The default week start is got
        from calendar.firstweekday(), and may be modified by
        calendar.setfirstweekday().
    :param count:
        If given, this determines how many occurrences will be generated.

        .. note::
            As of version 2.5.0, the use of the keyword ``until`` in conjunction
            with ``count`` is deprecated, to make sure ``dateutil`` is fully
            compliant with `RFC-5545 Sec. 3.3.10 <https://tools.ietf.org/
            html/rfc5545#section-3.3.10>`_. Therefore, ``until`` and ``count``
            **must not** occur in the same call to ``rrule``.
    :param until:
        If given, this must be a datetime instance specifying the upper-bound
        limit of the recurrence. The last recurrence in the rule is the greatest
        datetime that is less than or equal to the value specified in the
        ``until`` parameter.

        .. note::
            As of version 2.5.0, the use of the keyword ``until`` in conjunction
            with ``count`` is deprecated, to make sure ``dateutil`` is fully
            compliant with `RFC-5545 Sec. 3.3.10 <https://tools.ietf.org/
            html/rfc5545#section-3.3.10>`_. Therefore, ``until`` and ``count``
            **must not** occur in the same call to ``rrule``.
    :param bysetpos:
        If given, it must be either an integer, or a sequence of integers,
        positive or negative. Each given integer will specify an occurrence
        number, corresponding to the nth occurrence of the rule inside the
        frequency period. For example, a bysetpos of -1 if combined with a
        MONTHLY frequency, and a byweekday of (MO, TU, WE, TH, FR), will
        result in the last work day of every month.
    :param bymonth:
        If given, it must be either an integer, or a sequence of integers,
        meaning the months to apply the recurrence to.
    :param bymonthday:
        If given, it must be either an integer, or a sequence of integers,
        meaning the month days to apply the recurrence to.
    :param byyearday:
        If given, it must be either an integer, or a sequence of integers,
        meaning the year days to apply the recurrence to.
    :param byeaster:
        If given, it must be either an integer, or a sequence of integers,
        positive or negative. Each integer will define an offset from the
        Easter Sunday. Passing the offset 0 to byeaster will yield the Easter
        Sunday itself. This is an extension to the RFC specification.
    :param byweekno:
        If given, it must be either an integer, or a sequence of integers,
        meaning the week numbers to apply the recurrence to. Week numbers
        have the meaning described in ISO8601, that is, the first week of
        the year is that containing at least four days of the new year.
    :param byweekday:
        If given, it must be either an integer (0 == MO), a sequence of
        integers, one of the weekday constants (MO, TU, etc), or a sequence
        of these constants. When given, these variables will define the
        weekdays where the recurrence will be applied. It's also possible to
        use an argument n for the weekday instances, which will mean the nth
        occurrence of this weekday in the period. For example, with MONTHLY,
        or with YEARLY and BYMONTH, using FR(+1) in byweekday will specify the
        first friday of the month where the recurrence happens. Notice that in
        the RFC documentation, this is specified as BYDAY, but was renamed to
        avoid the ambiguity of that keyword.
    :param byhour:
        If given, it must be either an integer, or a sequence of integers,
        meaning the hours to apply the recurrence to.
    :param byminute:
        If given, it must be either an integer, or a sequence of integers,
        meaning the minutes to apply the recurrence to.
    :param bysecond:
        If given, it must be either an integer, or a sequence of integers,
        meaning the seconds to apply the recurrence to.
    :param cache:
        If given, it must be a boolean value specifying to enable or disable
        caching of results. If you will use the same rrule instance multiple
        times, enabling caching will improve the performance considerably.
     """

    def __init__(self, freq, dtstart=None, interval=1, wkst=None, count=None, until=None, bysetpos=None, bymonth=None, bymonthday=None, byyearday=None, byeaster=None, byweekno=None, byweekday=None, byhour=None, byminute=None, bysecond=None, cache=False):
        super(rrule, self).__init__(cache)
        global easter
        if not dtstart:
            if until and until.tzinfo:
                dtstart = datetime.datetime.now(tz=until.tzinfo).replace(microsecond=0)
            else:
                dtstart = datetime.datetime.now().replace(microsecond=0)
        elif not isinstance(dtstart, datetime.datetime):
            dtstart = datetime.datetime.fromordinal(dtstart.toordinal())
        else:
            dtstart = dtstart.replace(microsecond=0)
        self._dtstart = dtstart
        self._tzinfo = dtstart.tzinfo
        self._freq = freq
        self._interval = interval
        self._count = count
        self._original_rule = {}
        if until and (not isinstance(until, datetime.datetime)):
            until = datetime.datetime.fromordinal(until.toordinal())
        self._until = until
        if self._dtstart and self._until:
            if (self._dtstart.tzinfo is not None) != (self._until.tzinfo is not None):
                raise ValueError('RRULE UNTIL values must be specified in UTC when DTSTART is timezone-aware')
        if count is not None and until:
            warn("Using both 'count' and 'until' is inconsistent with RFC 5545 and has been deprecated in dateutil. Future versions will raise an error.", DeprecationWarning)
        if wkst is None:
            self._wkst = calendar.firstweekday()
        elif isinstance(wkst, integer_types):
            self._wkst = wkst
        else:
            self._wkst = wkst.weekday
        if bysetpos is None:
            self._bysetpos = None
        elif isinstance(bysetpos, integer_types):
            if bysetpos == 0 or not -366 <= bysetpos <= 366:
                raise ValueError('bysetpos must be between 1 and 366, or between -366 and -1')
            self._bysetpos = (bysetpos,)
        else:
            self._bysetpos = tuple(bysetpos)
            for pos in self._bysetpos:
                if pos == 0 or not -366 <= pos <= 366:
                    raise ValueError('bysetpos must be between 1 and 366, or between -366 and -1')
        if self._bysetpos:
            self._original_rule['bysetpos'] = self._bysetpos
        if byweekno is None and byyearday is None and (bymonthday is None) and (byweekday is None) and (byeaster is None):
            if freq == YEARLY:
                if bymonth is None:
                    bymonth = dtstart.month
                    self._original_rule['bymonth'] = None
                bymonthday = dtstart.day
                self._original_rule['bymonthday'] = None
            elif freq == MONTHLY:
                bymonthday = dtstart.day
                self._original_rule['bymonthday'] = None
            elif freq == WEEKLY:
                byweekday = dtstart.weekday()
                self._original_rule['byweekday'] = None
        if bymonth is None:
            self._bymonth = None
        else:
            if isinstance(bymonth, integer_types):
                bymonth = (bymonth,)
            self._bymonth = tuple(sorted(set(bymonth)))
            if 'bymonth' not in self._original_rule:
                self._original_rule['bymonth'] = self._bymonth
        if byyearday is None:
            self._byyearday = None
        else:
            if isinstance(byyearday, integer_types):
                byyearday = (byyearday,)
            self._byyearday = tuple(sorted(set(byyearday)))
            self._original_rule['byyearday'] = self._byyearday
        if byeaster is not None:
            if not easter:
                from dateutil import easter
            if isinstance(byeaster, integer_types):
                self._byeaster = (byeaster,)
            else:
                self._byeaster = tuple(sorted(byeaster))
            self._original_rule['byeaster'] = self._byeaster
        else:
            self._byeaster = None
        if bymonthday is None:
            self._bymonthday = ()
            self._bynmonthday = ()
        else:
            if isinstance(bymonthday, integer_types):
                bymonthday = (bymonthday,)
            bymonthday = set(bymonthday)
            self._bymonthday = tuple(sorted((x for x in bymonthday if x > 0)))
            self._bynmonthday = tuple(sorted((x for x in bymonthday if x < 0)))
            if 'bymonthday' not in self._original_rule:
                self._original_rule['bymonthday'] = tuple(itertools.chain(self._bymonthday, self._bynmonthday))
        if byweekno is None:
            self._byweekno = None
        else:
            if isinstance(byweekno, integer_types):
                byweekno = (byweekno,)
            self._byweekno = tuple(sorted(set(byweekno)))
            self._original_rule['byweekno'] = self._byweekno
        if byweekday is None:
            self._byweekday = None
            self._bynweekday = None
        else:
            if isinstance(byweekday, integer_types) or hasattr(byweekday, 'n'):
                byweekday = (byweekday,)
            self._byweekday = set()
            self._bynweekday = set()
            for wday in byweekday:
                if isinstance(wday, integer_types):
                    self._byweekday.add(wday)
                elif not wday.n or freq > MONTHLY:
                    self._byweekday.add(wday.weekday)
                else:
                    self._bynweekday.add((wday.weekday, wday.n))
            if not self._byweekday:
                self._byweekday = None
            elif not self._bynweekday:
                self._bynweekday = None
            if self._byweekday is not None:
                self._byweekday = tuple(sorted(self._byweekday))
                orig_byweekday = [weekday(x) for x in self._byweekday]
            else:
                orig_byweekday = ()
            if self._bynweekday is not None:
                self._bynweekday = tuple(sorted(self._bynweekday))
                orig_bynweekday = [weekday(*x) for x in self._bynweekday]
            else:
                orig_bynweekday = ()
            if 'byweekday' not in self._original_rule:
                self._original_rule['byweekday'] = tuple(itertools.chain(orig_byweekday, orig_bynweekday))
        if byhour is None:
            if freq < HOURLY:
                self._byhour = {dtstart.hour}
            else:
                self._byhour = None
        else:
            if isinstance(byhour, integer_types):
                byhour = (byhour,)
            if freq == HOURLY:
                self._byhour = self.__construct_byset(start=dtstart.hour, byxxx=byhour, base=24)
            else:
                self._byhour = set(byhour)
            self._byhour = tuple(sorted(self._byhour))
            self._original_rule['byhour'] = self._byhour
        if byminute is None:
            if freq < MINUTELY:
                self._byminute = {dtstart.minute}
            else:
                self._byminute = None
        else:
            if isinstance(byminute, integer_types):
                byminute = (byminute,)
            if freq == MINUTELY:
                self._byminute = self.__construct_byset(start=dtstart.minute, byxxx=byminute, base=60)
            else:
                self._byminute = set(byminute)
            self._byminute = tuple(sorted(self._byminute))
            self._original_rule['byminute'] = self._byminute
        if bysecond is None:
            if freq < SECONDLY:
                self._bysecond = (dtstart.second,)
            else:
                self._bysecond = None
        else:
            if isinstance(bysecond, integer_types):
                bysecond = (bysecond,)
            self._bysecond = set(bysecond)
            if freq == SECONDLY:
                self._bysecond = self.__construct_byset(start=dtstart.second, byxxx=bysecond, base=60)
            else:
                self._bysecond = set(bysecond)
            self._bysecond = tuple(sorted(self._bysecond))
            self._original_rule['bysecond'] = self._bysecond
        if self._freq >= HOURLY:
            self._timeset = None
        else:
            self._timeset = []
            for hour in self._byhour:
                for minute in self._byminute:
                    for second in self._bysecond:
                        self._timeset.append(datetime.time(hour, minute, second, tzinfo=self._tzinfo))
            self._timeset.sort()
            self._timeset = tuple(self._timeset)

    def __str__(self):
        """
        Output a string that would generate this RRULE if passed to rrulestr.
        This is mostly compatible with RFC5545, except for the
        dateutil-specific extension BYEASTER.
        """
        output = []
        h, m, s = [None] * 3
        if self._dtstart:
            output.append(self._dtstart.strftime('DTSTART:%Y%m%dT%H%M%S'))
            h, m, s = self._dtstart.timetuple()[3:6]
        parts = ['FREQ=' + FREQNAMES[self._freq]]
        if self._interval != 1:
            parts.append('INTERVAL=' + str(self._interval))
        if self._wkst:
            parts.append('WKST=' + repr(weekday(self._wkst))[0:2])
        if self._count is not None:
            parts.append('COUNT=' + str(self._count))
        if self._until:
            parts.append(self._until.strftime('UNTIL=%Y%m%dT%H%M%S'))
        if self._original_rule.get('byweekday') is not None:
            original_rule = dict(self._original_rule)
            wday_strings = []
            for wday in original_rule['byweekday']:
                if wday.n:
                    wday_strings.append('{n:+d}{wday}'.format(n=wday.n, wday=repr(wday)[0:2]))
                else:
                    wday_strings.append(repr(wday))
            original_rule['byweekday'] = wday_strings
        else:
            original_rule = self._original_rule
        partfmt = '{name}={vals}'
        for name, key in [('BYSETPOS', 'bysetpos'), ('BYMONTH', 'bymonth'), ('BYMONTHDAY', 'bymonthday'), ('BYYEARDAY', 'byyearday'), ('BYWEEKNO', 'byweekno'), ('BYDAY', 'byweekday'), ('BYHOUR', 'byhour'), ('BYMINUTE', 'byminute'), ('BYSECOND', 'bysecond'), ('BYEASTER', 'byeaster')]:
            value = original_rule.get(key)
            if value:
                parts.append(partfmt.format(name=name, vals=','.join((str(v) for v in value))))
        output.append('RRULE:' + ';'.join(parts))
        return '\n'.join(output)

    def replace(self, **kwargs):
        """Return new rrule with same attributes except for those attributes given new
           values by whichever keyword arguments are specified."""
        new_kwargs = {'interval': self._interval, 'count': self._count, 'dtstart': self._dtstart, 'freq': self._freq, 'until': self._until, 'wkst': self._wkst, 'cache': False if self._cache is None else True}
        new_kwargs.update(self._original_rule)
        new_kwargs.update(kwargs)
        return rrule(**new_kwargs)

    def _iter(self):
        year, month, day, hour, minute, second, weekday, yearday, _ = self._dtstart.timetuple()
        freq = self._freq
        interval = self._interval
        wkst = self._wkst
        until = self._until
        bymonth = self._bymonth
        byweekno = self._byweekno
        byyearday = self._byyearday
        byweekday = self._byweekday
        byeaster = self._byeaster
        bymonthday = self._bymonthday
        bynmonthday = self._bynmonthday
        bysetpos = self._bysetpos
        byhour = self._byhour
        byminute = self._byminute
        bysecond = self._bysecond
        ii = _iterinfo(self)
        ii.rebuild(year, month)
        getdayset = {YEARLY: ii.ydayset, MONTHLY: ii.mdayset, WEEKLY: ii.wdayset, DAILY: ii.ddayset, HOURLY: ii.ddayset, MINUTELY: ii.ddayset, SECONDLY: ii.ddayset}[freq]
        if freq < HOURLY:
            timeset = self._timeset
        else:
            gettimeset = {HOURLY: ii.htimeset, MINUTELY: ii.mtimeset, SECONDLY: ii.stimeset}[freq]
            if freq >= HOURLY and self._byhour and (hour not in self._byhour) or (freq >= MINUTELY and self._byminute and (minute not in self._byminute)) or (freq >= SECONDLY and self._bysecond and (second not in self._bysecond)):
                timeset = ()
            else:
                timeset = gettimeset(hour, minute, second)
        total = 0
        count = self._count
        while True:
            dayset, start, end = getdayset(year, month, day)
            filtered = False
            for i in dayset[start:end]:
                if bymonth and ii.mmask[i] not in bymonth or (byweekno and (not ii.wnomask[i])) or (byweekday and ii.wdaymask[i] not in byweekday) or (ii.nwdaymask and (not ii.nwdaymask[i])) or (byeaster and (not ii.eastermask[i])) or ((bymonthday or bynmonthday) and ii.mdaymask[i] not in bymonthday and (ii.nmdaymask[i] not in bynmonthday)) or (byyearday and (i < ii.yearlen and i + 1 not in byyearday and (-ii.yearlen + i not in byyearday) or (i >= ii.yearlen and i + 1 - ii.yearlen not in byyearday and (-ii.nextyearlen + i - ii.yearlen not in byyearday)))):
                    dayset[i] = None
                    filtered = True
            if bysetpos and timeset:
                poslist = []
                for pos in bysetpos:
                    if pos < 0:
                        daypos, timepos = divmod(pos, len(timeset))
                    else:
                        daypos, timepos = divmod(pos - 1, len(timeset))
                    try:
                        i = [x for x in dayset[start:end] if x is not None][daypos]
                        time = timeset[timepos]
                    except IndexError:
                        pass
                    else:
                        date = datetime.date.fromordinal(ii.yearordinal + i)
                        res = datetime.datetime.combine(date, time)
                        if res not in poslist:
                            poslist.append(res)
                poslist.sort()
                for res in poslist:
                    if until and res > until:
                        self._len = total
                        return
                    elif res >= self._dtstart:
                        if count is not None:
                            count -= 1
                            if count < 0:
                                self._len = total
                                return
                        total += 1
                        yield res
            else:
                for i in dayset[start:end]:
                    if i is not None:
                        date = datetime.date.fromordinal(ii.yearordinal + i)
                        for time in timeset:
                            res = datetime.datetime.combine(date, time)
                            if until and res > until:
                                self._len = total
                                return
                            elif res >= self._dtstart:
                                if count is not None:
                                    count -= 1
                                    if count < 0:
                                        self._len = total
                                        return
                                total += 1
                                yield res
            fixday = False
            if freq == YEARLY:
                year += interval
                if year > datetime.MAXYEAR:
                    self._len = total
                    return
                ii.rebuild(year, month)
            elif freq == MONTHLY:
                month += interval
                if month > 12:
                    div, mod = divmod(month, 12)
                    month = mod
                    year += div
                    if month == 0:
                        month = 12
                        year -= 1
                    if year > datetime.MAXYEAR:
                        self._len = total
                        return
                ii.rebuild(year, month)
            elif freq == WEEKLY:
                if wkst > weekday:
                    day += -(weekday + 1 + (6 - wkst)) + self._interval * 7
                else:
                    day += -(weekday - wkst) + self._interval * 7
                weekday = wkst
                fixday = True
            elif freq == DAILY:
                day += interval
                fixday = True
            elif freq == HOURLY:
                if filtered:
                    hour += (23 - hour) // interval * interval
                if byhour:
                    ndays, hour = self.__mod_distance(value=hour, byxxx=self._byhour, base=24)
                else:
                    ndays, hour = divmod(hour + interval, 24)
                if ndays:
                    day += ndays
                    fixday = True
                timeset = gettimeset(hour, minute, second)
            elif freq == MINUTELY:
                if filtered:
                    minute += (1439 - (hour * 60 + minute)) // interval * interval
                valid = False
                rep_rate = 24 * 60
                for j in range(rep_rate // gcd(interval, rep_rate)):
                    if byminute:
                        nhours, minute = self.__mod_distance(value=minute, byxxx=self._byminute, base=60)
                    else:
                        nhours, minute = divmod(minute + interval, 60)
                    div, hour = divmod(hour + nhours, 24)
                    if div:
                        day += div
                        fixday = True
                        filtered = False
                    if not byhour or hour in byhour:
                        valid = True
                        break
                if not valid:
                    raise ValueError('Invalid combination of interval and ' + 'byhour resulting in empty rule.')
                timeset = gettimeset(hour, minute, second)
            elif freq == SECONDLY:
                if filtered:
                    second += (86399 - (hour * 3600 + minute * 60 + second)) // interval * interval
                rep_rate = 24 * 3600
                valid = False
                for j in range(0, rep_rate // gcd(interval, rep_rate)):
                    if bysecond:
                        nminutes, second = self.__mod_distance(value=second, byxxx=self._bysecond, base=60)
                    else:
                        nminutes, second = divmod(second + interval, 60)
                    div, minute = divmod(minute + nminutes, 60)
                    if div:
                        hour += div
                        div, hour = divmod(hour, 24)
                        if div:
                            day += div
                            fixday = True
                    if (not byhour or hour in byhour) and (not byminute or minute in byminute) and (not bysecond or second in bysecond):
                        valid = True
                        break
                if not valid:
                    raise ValueError('Invalid combination of interval, ' + 'byhour and byminute resulting in empty' + ' rule.')
                timeset = gettimeset(hour, minute, second)
            if fixday and day > 28:
                daysinmonth = calendar.monthrange(year, month)[1]
                if day > daysinmonth:
                    while day > daysinmonth:
                        day -= daysinmonth
                        month += 1
                        if month == 13:
                            month = 1
                            year += 1
                            if year > datetime.MAXYEAR:
                                self._len = total
                                return
                        daysinmonth = calendar.monthrange(year, month)[1]
                    ii.rebuild(year, month)

    def __construct_byset(self, start, byxxx, base):
        """
        If a `BYXXX` sequence is passed to the constructor at the same level as
        `FREQ` (e.g. `FREQ=HOURLY,BYHOUR={2,4,7},INTERVAL=3`), there are some
        specifications which cannot be reached given some starting conditions.

        This occurs whenever the interval is not coprime with the base of a
        given unit and the difference between the starting position and the
        ending position is not coprime with the greatest common denominator
        between the interval and the base. For example, with a FREQ of hourly
        starting at 17:00 and an interval of 4, the only valid values for
        BYHOUR would be {21, 1, 5, 9, 13, 17}, because 4 and 24 are not
        coprime.

        :param start:
            Specifies the starting position.
        :param byxxx:
            An iterable containing the list of allowed values.
        :param base:
            The largest allowable value for the specified frequency (e.g.
            24 hours, 60 minutes).

        This does not preserve the type of the iterable, returning a set, since
        the values should be unique and the order is irrelevant, this will
        speed up later lookups.

        In the event of an empty set, raises a :exception:`ValueError`, as this
        results in an empty rrule.
        """
        cset = set()
        if isinstance(byxxx, integer_types):
            byxxx = (byxxx,)
        for num in byxxx:
            i_gcd = gcd(self._interval, base)
            if i_gcd == 1 or divmod(num - start, i_gcd)[1] == 0:
                cset.add(num)
        if len(cset) == 0:
            raise ValueError('Invalid rrule byxxx generates an empty set.')
        return cset

    def __mod_distance(self, value, byxxx, base):
        """
        Calculates the next value in a sequence where the `FREQ` parameter is
        specified along with a `BYXXX` parameter at the same "level"
        (e.g. `HOURLY` specified with `BYHOUR`).

        :param value:
            The old value of the component.
        :param byxxx:
            The `BYXXX` set, which should have been generated by
            `rrule._construct_byset`, or something else which checks that a
            valid rule is present.
        :param base:
            The largest allowable value for the specified frequency (e.g.
            24 hours, 60 minutes).

        If a valid value is not found after `base` iterations (the maximum
        number before the sequence would start to repeat), this raises a
        :exception:`ValueError`, as no valid values were found.

        This returns a tuple of `divmod(n*interval, base)`, where `n` is the
        smallest number of `interval` repetitions until the next specified
        value in `byxxx` is found.
        """
        accumulator = 0
        for ii in range(1, base + 1):
            div, value = divmod(value + self._interval, base)
            accumulator += div
            if value in byxxx:
                return (accumulator, value)