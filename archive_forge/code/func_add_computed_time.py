from datetime import datetime, time, timedelta
import pyparsing as pp
import calendar
from_ = CK("from").setParseAction(pp.replaceWith(1))
def add_computed_time(t):
    if t[0] in 'now noon midnight'.split():
        t['computed_time'] = {'now': datetime.now().time().replace(microsecond=0), 'noon': time(hour=12), 'midnight': time()}[t[0]]
    else:
        t['HH'] = {'am': int(t['HH']) % 12, 'pm': int(t['HH']) % 12 + 12}[t.ampm]
        t['computed_time'] = time(hour=t.HH, minute=t.MM, second=t.SS)