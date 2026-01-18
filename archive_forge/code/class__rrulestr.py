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
class _rrulestr(object):
    """ Parses a string representation of a recurrence rule or set of
    recurrence rules.

    :param s:
        Required, a string defining one or more recurrence rules.

    :param dtstart:
        If given, used as the default recurrence start if not specified in the
        rule string.

    :param cache:
        If set ``True`` caching of results will be enabled, improving
        performance of multiple queries considerably.

    :param unfold:
        If set ``True`` indicates that a rule string is split over more
        than one line and should be joined before processing.

    :param forceset:
        If set ``True`` forces a :class:`dateutil.rrule.rruleset` to
        be returned.

    :param compatible:
        If set ``True`` forces ``unfold`` and ``forceset`` to be ``True``.

    :param ignoretz:
        If set ``True``, time zones in parsed strings are ignored and a naive
        :class:`datetime.datetime` object is returned.

    :param tzids:
        If given, a callable or mapping used to retrieve a
        :class:`datetime.tzinfo` from a string representation.
        Defaults to :func:`dateutil.tz.gettz`.

    :param tzinfos:
        Additional time zone names / aliases which may be present in a string
        representation.  See :func:`dateutil.parser.parse` for more
        information.

    :return:
        Returns a :class:`dateutil.rrule.rruleset` or
        :class:`dateutil.rrule.rrule`
    """
    _freq_map = {'YEARLY': YEARLY, 'MONTHLY': MONTHLY, 'WEEKLY': WEEKLY, 'DAILY': DAILY, 'HOURLY': HOURLY, 'MINUTELY': MINUTELY, 'SECONDLY': SECONDLY}
    _weekday_map = {'MO': 0, 'TU': 1, 'WE': 2, 'TH': 3, 'FR': 4, 'SA': 5, 'SU': 6}

    def _handle_int(self, rrkwargs, name, value, **kwargs):
        rrkwargs[name.lower()] = int(value)

    def _handle_int_list(self, rrkwargs, name, value, **kwargs):
        rrkwargs[name.lower()] = [int(x) for x in value.split(',')]
    _handle_INTERVAL = _handle_int
    _handle_COUNT = _handle_int
    _handle_BYSETPOS = _handle_int_list
    _handle_BYMONTH = _handle_int_list
    _handle_BYMONTHDAY = _handle_int_list
    _handle_BYYEARDAY = _handle_int_list
    _handle_BYEASTER = _handle_int_list
    _handle_BYWEEKNO = _handle_int_list
    _handle_BYHOUR = _handle_int_list
    _handle_BYMINUTE = _handle_int_list
    _handle_BYSECOND = _handle_int_list

    def _handle_FREQ(self, rrkwargs, name, value, **kwargs):
        rrkwargs['freq'] = self._freq_map[value]

    def _handle_UNTIL(self, rrkwargs, name, value, **kwargs):
        global parser
        if not parser:
            from dateutil import parser
        try:
            rrkwargs['until'] = parser.parse(value, ignoretz=kwargs.get('ignoretz'), tzinfos=kwargs.get('tzinfos'))
        except ValueError:
            raise ValueError('invalid until date')

    def _handle_WKST(self, rrkwargs, name, value, **kwargs):
        rrkwargs['wkst'] = self._weekday_map[value]

    def _handle_BYWEEKDAY(self, rrkwargs, name, value, **kwargs):
        """
        Two ways to specify this: +1MO or MO(+1)
        """
        l = []
        for wday in value.split(','):
            if '(' in wday:
                splt = wday.split('(')
                w = splt[0]
                n = int(splt[1][:-1])
            elif len(wday):
                for i in range(len(wday)):
                    if wday[i] not in '+-0123456789':
                        break
                n = wday[:i] or None
                w = wday[i:]
                if n:
                    n = int(n)
            else:
                raise ValueError('Invalid (empty) BYDAY specification.')
            l.append(weekdays[self._weekday_map[w]](n))
        rrkwargs['byweekday'] = l
    _handle_BYDAY = _handle_BYWEEKDAY

    def _parse_rfc_rrule(self, line, dtstart=None, cache=False, ignoretz=False, tzinfos=None):
        if line.find(':') != -1:
            name, value = line.split(':')
            if name != 'RRULE':
                raise ValueError('unknown parameter name')
        else:
            value = line
        rrkwargs = {}
        for pair in value.split(';'):
            name, value = pair.split('=')
            name = name.upper()
            value = value.upper()
            try:
                getattr(self, '_handle_' + name)(rrkwargs, name, value, ignoretz=ignoretz, tzinfos=tzinfos)
            except AttributeError:
                raise ValueError("unknown parameter '%s'" % name)
            except (KeyError, ValueError):
                raise ValueError("invalid '%s': %s" % (name, value))
        return rrule(dtstart=dtstart, cache=cache, **rrkwargs)

    def _parse_date_value(self, date_value, parms, rule_tzids, ignoretz, tzids, tzinfos):
        global parser
        if not parser:
            from dateutil import parser
        datevals = []
        value_found = False
        TZID = None
        for parm in parms:
            if parm.startswith('TZID='):
                try:
                    tzkey = rule_tzids[parm.split('TZID=')[-1]]
                except KeyError:
                    continue
                if tzids is None:
                    from . import tz
                    tzlookup = tz.gettz
                elif callable(tzids):
                    tzlookup = tzids
                else:
                    tzlookup = getattr(tzids, 'get', None)
                    if tzlookup is None:
                        msg = 'tzids must be a callable, mapping, or None, not %s' % tzids
                        raise ValueError(msg)
                TZID = tzlookup(tzkey)
                continue
            if parm not in {'VALUE=DATE-TIME', 'VALUE=DATE'}:
                raise ValueError('unsupported parm: ' + parm)
            else:
                if value_found:
                    msg = 'Duplicate value parameter found in: ' + parm
                    raise ValueError(msg)
                value_found = True
        for datestr in date_value.split(','):
            date = parser.parse(datestr, ignoretz=ignoretz, tzinfos=tzinfos)
            if TZID is not None:
                if date.tzinfo is None:
                    date = date.replace(tzinfo=TZID)
                else:
                    raise ValueError('DTSTART/EXDATE specifies multiple timezone')
            datevals.append(date)
        return datevals

    def _parse_rfc(self, s, dtstart=None, cache=False, unfold=False, forceset=False, compatible=False, ignoretz=False, tzids=None, tzinfos=None):
        global parser
        if compatible:
            forceset = True
            unfold = True
        TZID_NAMES = dict(map(lambda x: (x.upper(), x), re.findall('TZID=(?P<name>[^:]+):', s)))
        s = s.upper()
        if not s.strip():
            raise ValueError('empty string')
        if unfold:
            lines = s.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].rstrip()
                if not line:
                    del lines[i]
                elif i > 0 and line[0] == ' ':
                    lines[i - 1] += line[1:]
                    del lines[i]
                else:
                    i += 1
        else:
            lines = s.split()
        if not forceset and len(lines) == 1 and (s.find(':') == -1 or s.startswith('RRULE:')):
            return self._parse_rfc_rrule(lines[0], cache=cache, dtstart=dtstart, ignoretz=ignoretz, tzinfos=tzinfos)
        else:
            rrulevals = []
            rdatevals = []
            exrulevals = []
            exdatevals = []
            for line in lines:
                if not line:
                    continue
                if line.find(':') == -1:
                    name = 'RRULE'
                    value = line
                else:
                    name, value = line.split(':', 1)
                parms = name.split(';')
                if not parms:
                    raise ValueError('empty property name')
                name = parms[0]
                parms = parms[1:]
                if name == 'RRULE':
                    for parm in parms:
                        raise ValueError('unsupported RRULE parm: ' + parm)
                    rrulevals.append(value)
                elif name == 'RDATE':
                    for parm in parms:
                        if parm != 'VALUE=DATE-TIME':
                            raise ValueError('unsupported RDATE parm: ' + parm)
                    rdatevals.append(value)
                elif name == 'EXRULE':
                    for parm in parms:
                        raise ValueError('unsupported EXRULE parm: ' + parm)
                    exrulevals.append(value)
                elif name == 'EXDATE':
                    exdatevals.extend(self._parse_date_value(value, parms, TZID_NAMES, ignoretz, tzids, tzinfos))
                elif name == 'DTSTART':
                    dtvals = self._parse_date_value(value, parms, TZID_NAMES, ignoretz, tzids, tzinfos)
                    if len(dtvals) != 1:
                        raise ValueError('Multiple DTSTART values specified:' + value)
                    dtstart = dtvals[0]
                else:
                    raise ValueError('unsupported property: ' + name)
            if forceset or len(rrulevals) > 1 or rdatevals or exrulevals or exdatevals:
                if not parser and (rdatevals or exdatevals):
                    from dateutil import parser
                rset = rruleset(cache=cache)
                for value in rrulevals:
                    rset.rrule(self._parse_rfc_rrule(value, dtstart=dtstart, ignoretz=ignoretz, tzinfos=tzinfos))
                for value in rdatevals:
                    for datestr in value.split(','):
                        rset.rdate(parser.parse(datestr, ignoretz=ignoretz, tzinfos=tzinfos))
                for value in exrulevals:
                    rset.exrule(self._parse_rfc_rrule(value, dtstart=dtstart, ignoretz=ignoretz, tzinfos=tzinfos))
                for value in exdatevals:
                    rset.exdate(value)
                if compatible and dtstart:
                    rset.rdate(dtstart)
                return rset
            else:
                return self._parse_rfc_rrule(rrulevals[0], dtstart=dtstart, cache=cache, ignoretz=ignoretz, tzinfos=tzinfos)

    def __call__(self, s, **kwargs):
        return self._parse_rfc(s, **kwargs)