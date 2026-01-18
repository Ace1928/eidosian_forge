from suds import UnicodeMixin
import datetime
import re
import time
def _date_from_match(match_object):
    """
    Create a date object from a regular expression match.

    The regular expression match is expected to be from _RE_DATE or
    _RE_DATETIME.

    @param match_object: The regular expression match.
    @type match_object: B{re}.I{MatchObject}
    @return: A date object.
    @rtype: B{datetime}.I{date}

    """
    year = int(match_object.group('year'))
    month = int(match_object.group('month'))
    day = int(match_object.group('day'))
    return datetime.date(year, month, day)