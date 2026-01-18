import sys
from math import trunc
from typing import (
class AzerbaijaniLocale(Locale):
    names = ['az', 'az-az']
    past = '{0} əvvəl'
    future = '{0} sonra'
    timeframes = {'now': 'indi', 'second': 'bir saniyə', 'seconds': '{0} saniyə', 'minute': 'bir dəqiqə', 'minutes': '{0} dəqiqə', 'hour': 'bir saat', 'hours': '{0} saat', 'day': 'bir gün', 'days': '{0} gün', 'week': 'bir həftə', 'weeks': '{0} həftə', 'month': 'bir ay', 'months': '{0} ay', 'year': 'bir il', 'years': '{0} il'}
    month_names = ['', 'Yanvar', 'Fevral', 'Mart', 'Aprel', 'May', 'İyun', 'İyul', 'Avqust', 'Sentyabr', 'Oktyabr', 'Noyabr', 'Dekabr']
    month_abbreviations = ['', 'Yan', 'Fev', 'Mar', 'Apr', 'May', 'İyn', 'İyl', 'Avq', 'Sen', 'Okt', 'Noy', 'Dek']
    day_names = ['', 'Bazar ertəsi', 'Çərşənbə axşamı', 'Çərşənbə', 'Cümə axşamı', 'Cümə', 'Şənbə', 'Bazar']
    day_abbreviations = ['', 'Ber', 'Çax', 'Çər', 'Cax', 'Cüm', 'Şnb', 'Bzr']