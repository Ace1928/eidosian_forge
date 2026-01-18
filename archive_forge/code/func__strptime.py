import time
import locale
import calendar
from re import compile as re_compile
from re import IGNORECASE
from re import escape as re_escape
from datetime import (date as datetime_date,
from _thread import allocate_lock as _thread_allocate_lock
def _strptime(data_string, format='%a %b %d %H:%M:%S %Y'):
    """Return a 2-tuple consisting of a time struct and an int containing
    the number of microseconds based on the input string and the
    format string."""
    for index, arg in enumerate([data_string, format]):
        if not isinstance(arg, str):
            msg = 'strptime() argument {} must be str, not {}'
            raise TypeError(msg.format(index, type(arg)))
    global _TimeRE_cache, _regex_cache
    with _cache_lock:
        locale_time = _TimeRE_cache.locale_time
        if _getlang() != locale_time.lang or time.tzname != locale_time.tzname or time.daylight != locale_time.daylight:
            _TimeRE_cache = TimeRE()
            _regex_cache.clear()
            locale_time = _TimeRE_cache.locale_time
        if len(_regex_cache) > _CACHE_MAX_SIZE:
            _regex_cache.clear()
        format_regex = _regex_cache.get(format)
        if not format_regex:
            try:
                format_regex = _TimeRE_cache.compile(format)
            except KeyError as err:
                bad_directive = err.args[0]
                if bad_directive == '\\':
                    bad_directive = '%'
                del err
                raise ValueError("'%s' is a bad directive in format '%s'" % (bad_directive, format)) from None
            except IndexError:
                raise ValueError("stray %% in format '%s'" % format) from None
            _regex_cache[format] = format_regex
    found = format_regex.match(data_string)
    if not found:
        raise ValueError('time data %r does not match format %r' % (data_string, format))
    if len(data_string) != found.end():
        raise ValueError('unconverted data remains: %s' % data_string[found.end():])
    iso_year = year = None
    month = day = 1
    hour = minute = second = fraction = 0
    tz = -1
    gmtoff = None
    gmtoff_fraction = 0
    iso_week = week_of_year = None
    week_of_year_start = None
    weekday = julian = None
    found_dict = found.groupdict()
    for group_key in found_dict.keys():
        if group_key == 'y':
            year = int(found_dict['y'])
            if year <= 68:
                year += 2000
            else:
                year += 1900
        elif group_key == 'Y':
            year = int(found_dict['Y'])
        elif group_key == 'G':
            iso_year = int(found_dict['G'])
        elif group_key == 'm':
            month = int(found_dict['m'])
        elif group_key == 'B':
            month = locale_time.f_month.index(found_dict['B'].lower())
        elif group_key == 'b':
            month = locale_time.a_month.index(found_dict['b'].lower())
        elif group_key == 'd':
            day = int(found_dict['d'])
        elif group_key == 'H':
            hour = int(found_dict['H'])
        elif group_key == 'I':
            hour = int(found_dict['I'])
            ampm = found_dict.get('p', '').lower()
            if ampm in ('', locale_time.am_pm[0]):
                if hour == 12:
                    hour = 0
            elif ampm == locale_time.am_pm[1]:
                if hour != 12:
                    hour += 12
        elif group_key == 'M':
            minute = int(found_dict['M'])
        elif group_key == 'S':
            second = int(found_dict['S'])
        elif group_key == 'f':
            s = found_dict['f']
            s += '0' * (6 - len(s))
            fraction = int(s)
        elif group_key == 'A':
            weekday = locale_time.f_weekday.index(found_dict['A'].lower())
        elif group_key == 'a':
            weekday = locale_time.a_weekday.index(found_dict['a'].lower())
        elif group_key == 'w':
            weekday = int(found_dict['w'])
            if weekday == 0:
                weekday = 6
            else:
                weekday -= 1
        elif group_key == 'u':
            weekday = int(found_dict['u'])
            weekday -= 1
        elif group_key == 'j':
            julian = int(found_dict['j'])
        elif group_key in ('U', 'W'):
            week_of_year = int(found_dict[group_key])
            if group_key == 'U':
                week_of_year_start = 6
            else:
                week_of_year_start = 0
        elif group_key == 'V':
            iso_week = int(found_dict['V'])
        elif group_key == 'z':
            z = found_dict['z']
            if z == 'Z':
                gmtoff = 0
            else:
                if z[3] == ':':
                    z = z[:3] + z[4:]
                    if len(z) > 5:
                        if z[5] != ':':
                            msg = f'Inconsistent use of : in {found_dict['z']}'
                            raise ValueError(msg)
                        z = z[:5] + z[6:]
                hours = int(z[1:3])
                minutes = int(z[3:5])
                seconds = int(z[5:7] or 0)
                gmtoff = hours * 60 * 60 + minutes * 60 + seconds
                gmtoff_remainder = z[8:]
                gmtoff_remainder_padding = '0' * (6 - len(gmtoff_remainder))
                gmtoff_fraction = int(gmtoff_remainder + gmtoff_remainder_padding)
                if z.startswith('-'):
                    gmtoff = -gmtoff
                    gmtoff_fraction = -gmtoff_fraction
        elif group_key == 'Z':
            found_zone = found_dict['Z'].lower()
            for value, tz_values in enumerate(locale_time.timezone):
                if found_zone in tz_values:
                    if time.tzname[0] == time.tzname[1] and time.daylight and (found_zone not in ('utc', 'gmt')):
                        break
                    else:
                        tz = value
                        break
    if iso_year is not None:
        if julian is not None:
            raise ValueError("Day of the year directive '%j' is not compatible with ISO year directive '%G'. Use '%Y' instead.")
        elif iso_week is None or weekday is None:
            raise ValueError("ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive ('%A', '%a', '%w', or '%u').")
    elif iso_week is not None:
        if year is None or weekday is None:
            raise ValueError("ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive ('%A', '%a', '%w', or '%u').")
        else:
            raise ValueError("ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.")
    leap_year_fix = False
    if year is None:
        if month == 2 and day == 29:
            year = 1904
            leap_year_fix = True
        else:
            year = 1900
    if julian is None and weekday is not None:
        if week_of_year is not None:
            week_starts_Mon = True if week_of_year_start == 0 else False
            julian = _calc_julian_from_U_or_W(year, week_of_year, weekday, week_starts_Mon)
        elif iso_year is not None and iso_week is not None:
            year, julian = _calc_julian_from_V(iso_year, iso_week, weekday + 1)
        if julian is not None and julian <= 0:
            year -= 1
            yday = 366 if calendar.isleap(year) else 365
            julian += yday
    if julian is None:
        julian = datetime_date(year, month, day).toordinal() - datetime_date(year, 1, 1).toordinal() + 1
    else:
        datetime_result = datetime_date.fromordinal(julian - 1 + datetime_date(year, 1, 1).toordinal())
        year = datetime_result.year
        month = datetime_result.month
        day = datetime_result.day
    if weekday is None:
        weekday = datetime_date(year, month, day).weekday()
    tzname = found_dict.get('Z')
    if leap_year_fix:
        year = 1900
    return ((year, month, day, hour, minute, second, weekday, julian, tz, tzname, gmtoff), fraction, gmtoff_fraction)