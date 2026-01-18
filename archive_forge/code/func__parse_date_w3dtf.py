import datetime
def _parse_date_w3dtf(datestr):
    if not datestr.strip():
        return None
    parts = datestr.lower().split('t')
    if len(parts) == 1:
        parts = parts[0].split()
        if len(parts) == 1:
            parts.append('00:00:00z')
    elif len(parts) > 2:
        return None
    date = parts[0].split('-', 2)
    if not date or len(date[0]) != 4:
        return None
    date.extend(['1'] * (3 - len(date)))
    try:
        year, month, day = [int(i) for i in date]
    except ValueError:
        return None
    if parts[1].endswith('z'):
        parts[1] = parts[1][:-1]
        parts.append('z')
    loc = parts[1].find('-') + 1 or parts[1].find('+') + 1 or len(parts[1]) + 1
    loc = loc - 1
    parts.append(parts[1][loc:])
    parts[1] = parts[1][:loc]
    time = parts[1].split(':', 2)
    time.extend(['0'] * (3 - len(time)))
    if parts[2][:1] in ('-', '+'):
        try:
            tzhour = int(parts[2][1:3])
            tzmin = int(parts[2][4:])
        except ValueError:
            return None
        if parts[2].startswith('-'):
            tzhour = tzhour * -1
            tzmin = tzmin * -1
    else:
        tzhour = timezonenames.get(parts[2], 0)
        tzmin = 0
    try:
        hour, minute, second = [int(float(i)) for i in time]
    except ValueError:
        return None
    try:
        stamp = datetime.datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None
    delta = datetime.timedelta(0, 0, 0, 0, tzmin, tzhour)
    try:
        return (stamp - delta).utctimetuple()
    except (OverflowError, ValueError):
        return None