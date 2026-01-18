from datetime import datetime, time, timedelta
import pyparsing as pp
import calendar
from_ = CK("from").setParseAction(pp.replaceWith(1))
def add_default_time_ref_fields(t):
    if 'time_delta' not in t:
        t['time_delta'] = timedelta()