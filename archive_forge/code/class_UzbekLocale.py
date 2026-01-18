import sys
from math import trunc
from typing import (
class UzbekLocale(Locale):
    names = ['uz', 'uz-uz']
    past = '{0}dan avval'
    future = '{0}dan keyin'
    timeframes = {'now': 'hozir', 'second': 'bir soniya', 'seconds': '{0} soniya', 'minute': 'bir daqiqa', 'minutes': '{0} daqiqa', 'hour': 'bir soat', 'hours': '{0} soat', 'day': 'bir kun', 'days': '{0} kun', 'week': 'bir hafta', 'weeks': '{0} hafta', 'month': 'bir oy', 'months': '{0} oy', 'year': 'bir yil', 'years': '{0} yil'}
    month_names = ['', 'Yanvar', 'Fevral', 'Mart', 'Aprel', 'May', 'Iyun', 'Iyul', 'Avgust', 'Sentyabr', 'Oktyabr', 'Noyabr', 'Dekabr']
    month_abbreviations = ['', 'Yan', 'Fev', 'Mar', 'Apr', 'May', 'Iyn', 'Iyl', 'Avg', 'Sen', 'Okt', 'Noy', 'Dek']
    day_names = ['', 'Dushanba', 'Seshanba', 'Chorshanba', 'Payshanba', 'Juma', 'Shanba', 'Yakshanba']
    day_abbreviations = ['', 'Dush', 'Sesh', 'Chor', 'Pay', 'Jum', 'Shan', 'Yak']