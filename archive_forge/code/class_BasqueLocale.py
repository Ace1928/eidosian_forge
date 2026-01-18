import sys
from math import trunc
from typing import (
class BasqueLocale(Locale):
    names = ['eu', 'eu-eu']
    past = 'duela {0}'
    future = '{0}'
    timeframes = {'now': 'Orain', 'second': 'segundo bat', 'seconds': '{0} segundu', 'minute': 'minutu bat', 'minutes': '{0} minutu', 'hour': 'ordu bat', 'hours': '{0} ordu', 'day': 'egun bat', 'days': '{0} egun', 'month': 'hilabete bat', 'months': '{0} hilabet', 'year': 'urte bat', 'years': '{0} urte'}
    month_names = ['', 'urtarrilak', 'otsailak', 'martxoak', 'apirilak', 'maiatzak', 'ekainak', 'uztailak', 'abuztuak', 'irailak', 'urriak', 'azaroak', 'abenduak']
    month_abbreviations = ['', 'urt', 'ots', 'mar', 'api', 'mai', 'eka', 'uzt', 'abu', 'ira', 'urr', 'aza', 'abe']
    day_names = ['', 'astelehena', 'asteartea', 'asteazkena', 'osteguna', 'ostirala', 'larunbata', 'igandea']
    day_abbreviations = ['', 'al', 'ar', 'az', 'og', 'ol', 'lr', 'ig']