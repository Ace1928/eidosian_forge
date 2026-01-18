from time import localtime
from datetime import date, datetime, time, timedelta
from MySQLdb._mysql import string_literal
def DateTimeDelta2literal(d, c):
    """Format a DateTimeDelta object as a time."""
    return string_literal(format_TIMEDELTA(d))