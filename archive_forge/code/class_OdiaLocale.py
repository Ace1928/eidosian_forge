import sys
from math import trunc
from typing import (
class OdiaLocale(Locale):
    names = ['or', 'or-in']
    past = '{0} ପୂର୍ବେ'
    future = '{0} ପରେ'
    timeframes = {'now': 'ବର୍ତ୍ତମାନ', 'second': 'ଏକ ସେକେଣ୍ଡ', 'seconds': '{0} ସେକେଣ୍ଡ', 'minute': 'ଏକ ମିନଟ', 'minutes': '{0} ମିନଟ', 'hour': 'ଏକ ଘଣ୍ଟା', 'hours': '{0} ଘଣ୍ଟା', 'day': 'ଏକ ଦିନ', 'days': '{0} ଦିନ', 'month': 'ଏକ ମାସ', 'months': '{0} ମାସ ', 'year': 'ଏକ ବର୍ଷ', 'years': '{0} ବର୍ଷ'}
    meridians = {'am': 'ପୂର୍ବାହ୍ନ', 'pm': 'ଅପରାହ୍ନ', 'AM': 'ପୂର୍ବାହ୍ନ', 'PM': 'ଅପରାହ୍ନ'}
    month_names = ['', 'ଜାନୁଆରୀ', 'ଫେବୃଆରୀ', 'ମାର୍ଚ୍ଚ୍', 'ଅପ୍ରେଲ', 'ମଇ', 'ଜୁନ୍', 'ଜୁଲାଇ', 'ଅଗଷ୍ଟ', 'ସେପ୍ଟେମ୍ବର', 'ଅକ୍ଟୋବର୍', 'ନଭେମ୍ବର୍', 'ଡିସେମ୍ବର୍']
    month_abbreviations = ['', 'ଜାନୁ', 'ଫେବୃ', 'ମାର୍ଚ୍ଚ୍', 'ଅପ୍ରେ', 'ମଇ', 'ଜୁନ୍', 'ଜୁଲା', 'ଅଗ', 'ସେପ୍ଟେ', 'ଅକ୍ଟୋ', 'ନଭେ', 'ଡିସେ']
    day_names = ['', 'ସୋମବାର', 'ମଙ୍ଗଳବାର', 'ବୁଧବାର', 'ଗୁରୁବାର', 'ଶୁକ୍ରବାର', 'ଶନିବାର', 'ରବିବାର']
    day_abbreviations = ['', 'ସୋମ', 'ମଙ୍ଗଳ', 'ବୁଧ', 'ଗୁରୁ', 'ଶୁକ୍ର', 'ଶନି', 'ରବି']

    def _ordinal_number(self, n: int) -> str:
        if n > 10 or n == 0:
            return f'{n}ତମ'
        if n in [1, 5, 7, 8, 9, 10]:
            return f'{n}ମ'
        if n in [2, 3]:
            return f'{n}ୟ'
        if n == 4:
            return f'{n}ର୍ଥ'
        if n == 6:
            return f'{n}ଷ୍ଠ'
        return ''