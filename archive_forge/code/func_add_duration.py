import calendar
from datetime import datetime
from datetime import timedelta
import re
import sys
import time
def add_duration(tid, duration):
    sign, dur = parse_duration(duration)
    if sign == '+':
        temp = tid.tm_mon + dur['tm_mon']
        month = modulo(temp, 1, 13)
        carry = f_quotient(temp, 1, 13)
        year = tid.tm_year + dur['tm_year'] + carry
        temp = tid.tm_sec + dur['tm_sec']
        secs = modulo(temp, 60)
        carry = f_quotient(temp, 60)
        temp = tid.tm_min + dur['tm_min'] + carry
        minutes = modulo(temp, 60)
        carry = f_quotient(temp, 60)
        temp = tid.tm_hour + dur['tm_hour'] + carry
        hour = modulo(temp, 24)
        carry = f_quotient(temp, 24)
        if tid.tm_mday > maximum_day_in_month_for(year, month):
            temp_days = maximum_day_in_month_for(year, month)
        elif tid.tm_mday < 1:
            temp_days = 1
        else:
            temp_days = tid.tm_mday
        days = temp_days + dur['tm_mday'] + carry
        while True:
            if days < 1:
                days = days + maximum_day_in_month_for(year, month - 1)
                carry = -1
            elif days > maximum_day_in_month_for(year, month):
                days -= maximum_day_in_month_for(year, month)
                carry = 1
            else:
                break
            temp = month + carry
            month = modulo(temp, 1, 13)
            year += f_quotient(temp, 1, 13)
        return time.localtime(time.mktime((year, month, days, hour, minutes, secs, 0, 0, -1)))
    else:
        pass