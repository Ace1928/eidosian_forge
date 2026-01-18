from time import localtime
from datetime import date, datetime, time, timedelta
from MySQLdb._mysql import string_literal
def DateTime_or_None(s):
    try:
        if len(s) < 11:
            return Date_or_None(s)
        micros = s[20:]
        if len(micros) == 0:
            micros = 0
        elif len(micros) < 7:
            micros = int(micros) * 10 ** (6 - len(micros))
        else:
            return None
        return datetime(int(s[:4]), int(s[5:7]), int(s[8:10]), int(s[11:13] or 0), int(s[14:16] or 0), int(s[17:19] or 0), micros)
    except ValueError:
        return None