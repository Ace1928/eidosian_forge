import datetime
def _parse_date_rfc822(date):
    """Parse RFC 822 dates and times
    http://tools.ietf.org/html/rfc822#section-5

    There are some formatting differences that are accounted for:
    1. Years may be two or four digits.
    2. The month and day can be swapped.
    3. Additional timezone names are supported.
    4. A default time and timezone are assumed if only a date is present.

    :param str date: a date/time string that will be converted to a time tuple
    :returns: a UTC time tuple, or None
    :rtype: time.struct_time | None
    """
    parts = date.lower().split()
    if len(parts) < 5:
        parts.extend(('00:00:00', '0000'))
    if parts[0][:3] in day_names:
        parts = parts[1:]
    if len(parts) < 5:
        return None
    month = months.get(parts[1][:3])
    try:
        day = int(parts[0])
    except ValueError:
        if months.get(parts[0][:3]):
            try:
                day = int(parts[1])
            except ValueError:
                return None
            month = months.get(parts[0][:3])
        else:
            return None
    if not month:
        return None
    try:
        year = int(parts[2])
    except ValueError:
        return None
    if len(parts[2]) <= 2:
        year += (1900, 2000)[year < 90]
    time_parts = parts[3].split(':')
    time_parts.extend(('0',) * (3 - len(time_parts)))
    try:
        hour, minute, second = [int(i) for i in time_parts]
    except ValueError:
        return None
    if parts[4].startswith('etc/'):
        parts[4] = parts[4][4:]
    if parts[4].startswith('gmt'):
        parts[4] = ''.join(parts[4][3:].split(':')) or 'gmt'
    if parts[4] and parts[4][0] in ('-', '+'):
        try:
            if ':' in parts[4]:
                timezone_hours = int(parts[4][1:3])
                timezone_minutes = int(parts[4][4:])
            else:
                timezone_hours = int(parts[4][1:3])
                timezone_minutes = int(parts[4][3:])
        except ValueError:
            return None
        if parts[4].startswith('-'):
            timezone_hours *= -1
            timezone_minutes *= -1
    else:
        timezone_hours = timezone_names.get(parts[4], 0)
        timezone_minutes = 0
    try:
        stamp = datetime.datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None
    delta = datetime.timedelta(0, 0, 0, 0, timezone_minutes, timezone_hours)
    try:
        return (stamp - delta).utctimetuple()
    except (OverflowError, ValueError):
        return None