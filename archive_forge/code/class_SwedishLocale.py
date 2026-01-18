import sys
from math import trunc
from typing import (
class SwedishLocale(Locale):
    names = ['sv', 'sv-se']
    past = 'för {0} sen'
    future = 'om {0}'
    and_word = 'och'
    timeframes = {'now': 'just nu', 'second': 'en sekund', 'seconds': '{0} sekunder', 'minute': 'en minut', 'minutes': '{0} minuter', 'hour': 'en timme', 'hours': '{0} timmar', 'day': 'en dag', 'days': '{0} dagar', 'week': 'en vecka', 'weeks': '{0} veckor', 'month': 'en månad', 'months': '{0} månader', 'year': 'ett år', 'years': '{0} år'}
    month_names = ['', 'januari', 'februari', 'mars', 'april', 'maj', 'juni', 'juli', 'augusti', 'september', 'oktober', 'november', 'december']
    month_abbreviations = ['', 'jan', 'feb', 'mar', 'apr', 'maj', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'dec']
    day_names = ['', 'måndag', 'tisdag', 'onsdag', 'torsdag', 'fredag', 'lördag', 'söndag']
    day_abbreviations = ['', 'mån', 'tis', 'ons', 'tor', 'fre', 'lör', 'sön']