import email._parseaddr
import re
import time
def _parse_date_perforce(date_string):
    """parse a date in yyyy/mm/dd hh:mm:ss TTT format"""
    _my_date_pattern = re.compile('(\\w{,3}), (\\d{,4})/(\\d{,2})/(\\d{2}) (\\d{,2}):(\\d{2}):(\\d{2}) (\\w{,3})')
    m = _my_date_pattern.search(date_string)
    if m is None:
        return None
    dow, year, month, day, hour, minute, second, tz = m.groups()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    new_date_string = '%s, %s %s %s %s:%s:%s %s' % (dow, day, months[int(month) - 1], year, hour, minute, second, tz)
    tm = email._parseaddr.parsedate_tz(new_date_string)
    if tm:
        return time.gmtime(email._parseaddr.mktime_tz(tm))