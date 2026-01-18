import re
from datetime import date, timedelta
from isodate.duration import Duration
from isodate.isotzinfo import tz_isoformat
def _strfdt(tdt, format, yeardigits=4):
    """
    this is the work method for time and date instances.

    see strftime for more details.
    """

    def repl(match):
        """
        lookup format command and return corresponding replacement.
        """
        if match.group(0) in STRF_DT_MAP:
            return STRF_DT_MAP[match.group(0)](tdt, yeardigits)
        return match.group(0)
    return re.sub('%d|%f|%H|%j|%m|%M|%S|%w|%W|%Y|%C|%z|%Z|%h|%%', repl, format)