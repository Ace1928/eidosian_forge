import sys
from math import trunc
from typing import (
class RomanianLocale(Locale):
    names = ['ro', 'ro-ro']
    past = '{0} în urmă'
    future = 'peste {0}'
    and_word = 'și'
    timeframes = {'now': 'acum', 'second': 'o secunda', 'seconds': '{0} câteva secunde', 'minute': 'un minut', 'minutes': '{0} minute', 'hour': 'o oră', 'hours': '{0} ore', 'day': 'o zi', 'days': '{0} zile', 'month': 'o lună', 'months': '{0} luni', 'year': 'un an', 'years': '{0} ani'}
    month_names = ['', 'ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie']
    month_abbreviations = ['', 'ian', 'febr', 'mart', 'apr', 'mai', 'iun', 'iul', 'aug', 'sept', 'oct', 'nov', 'dec']
    day_names = ['', 'luni', 'marți', 'miercuri', 'joi', 'vineri', 'sâmbătă', 'duminică']
    day_abbreviations = ['', 'Lun', 'Mar', 'Mie', 'Joi', 'Vin', 'Sâm', 'Dum']