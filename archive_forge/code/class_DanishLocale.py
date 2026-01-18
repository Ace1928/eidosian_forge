import sys
from math import trunc
from typing import (
class DanishLocale(Locale):
    names = ['da', 'da-dk']
    past = 'for {0} siden'
    future = 'om {0}'
    and_word = 'og'
    timeframes = {'now': 'lige nu', 'second': 'et sekund', 'seconds': '{0} sekunder', 'minute': 'et minut', 'minutes': '{0} minutter', 'hour': 'en time', 'hours': '{0} timer', 'day': 'en dag', 'days': '{0} dage', 'week': 'en uge', 'weeks': '{0} uger', 'month': 'en måned', 'months': '{0} måneder', 'year': 'et år', 'years': '{0} år'}
    month_names = ['', 'januar', 'februar', 'marts', 'april', 'maj', 'juni', 'juli', 'august', 'september', 'oktober', 'november', 'december']
    month_abbreviations = ['', 'jan', 'feb', 'mar', 'apr', 'maj', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'dec']
    day_names = ['', 'mandag', 'tirsdag', 'onsdag', 'torsdag', 'fredag', 'lørdag', 'søndag']
    day_abbreviations = ['', 'man', 'tir', 'ons', 'tor', 'fre', 'lør', 'søn']

    def _ordinal_number(self, n: int) -> str:
        return f'{n}.'