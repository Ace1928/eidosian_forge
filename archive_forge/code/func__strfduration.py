import re
from datetime import date, timedelta
from isodate.duration import Duration
from isodate.isotzinfo import tz_isoformat
def _strfduration(tdt, format, yeardigits=4):
    """
    this is the work method for timedelta and Duration instances.

    see strftime for more details.
    """

    def repl(match):
        """
        lookup format command and return corresponding replacement.
        """
        if match.group(0) in STRF_D_MAP:
            return STRF_D_MAP[match.group(0)](tdt, yeardigits)
        elif match.group(0) == '%P':
            ret = []
            if isinstance(tdt, Duration):
                if tdt.years:
                    ret.append('%sY' % abs(tdt.years))
                if tdt.months:
                    ret.append('%sM' % abs(tdt.months))
            usecs = abs((tdt.days * 24 * 60 * 60 + tdt.seconds) * 1000000 + tdt.microseconds)
            seconds, usecs = divmod(usecs, 1000000)
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            days, hours = divmod(hours, 24)
            if days:
                ret.append('%sD' % days)
            if hours or minutes or seconds or usecs:
                ret.append('T')
                if hours:
                    ret.append('%sH' % hours)
                if minutes:
                    ret.append('%sM' % minutes)
                if seconds or usecs:
                    if usecs:
                        ret.append(('%d.%06d' % (seconds, usecs)).rstrip('0'))
                    else:
                        ret.append('%d' % seconds)
                    ret.append('S')
            return ret and ''.join(ret) or '0D'
        elif match.group(0) == '%p':
            return str(abs(tdt.days // 7)) + 'W'
        return match.group(0)
    return re.sub('%d|%f|%H|%m|%M|%S|%W|%Y|%C|%%|%P|%p', repl, format)