import sys
from math import trunc
from typing import (
class NewNorwegianLocale(Locale):
    names = ['nn', 'nn-no']
    past = 'for {0} sidan'
    future = 'om {0}'
    timeframes = {'now': 'no nettopp', 'second': 'eitt sekund', 'seconds': '{0} sekund', 'minute': 'eitt minutt', 'minutes': '{0} minutt', 'hour': 'ein time', 'hours': '{0} timar', 'day': 'ein dag', 'days': '{0} dagar', 'week': 'ei veke', 'weeks': '{0} veker', 'month': 'ein månad', 'months': '{0} månader', 'year': 'eitt år', 'years': '{0} år'}
    month_names = ['', 'januar', 'februar', 'mars', 'april', 'mai', 'juni', 'juli', 'august', 'september', 'oktober', 'november', 'desember']
    month_abbreviations = ['', 'jan', 'feb', 'mar', 'apr', 'mai', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'des']
    day_names = ['', 'måndag', 'tysdag', 'onsdag', 'torsdag', 'fredag', 'laurdag', 'sundag']
    day_abbreviations = ['', 'må', 'ty', 'on', 'to', 'fr', 'la', 'su']

    def _ordinal_number(self, n: int) -> str:
        return f'{n}.'