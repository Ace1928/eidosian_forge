import sys
from math import trunc
from typing import (
class TurkishLocale(Locale):
    names = ['tr', 'tr-tr']
    past = '{0} önce'
    future = '{0} sonra'
    and_word = 've'
    timeframes = {'now': 'şimdi', 'second': 'bir saniye', 'seconds': '{0} saniye', 'minute': 'bir dakika', 'minutes': '{0} dakika', 'hour': 'bir saat', 'hours': '{0} saat', 'day': 'bir gün', 'days': '{0} gün', 'week': 'bir hafta', 'weeks': '{0} hafta', 'month': 'bir ay', 'months': '{0} ay', 'year': 'bir yıl', 'years': '{0} yıl'}
    meridians = {'am': 'öö', 'pm': 'ös', 'AM': 'ÖÖ', 'PM': 'ÖS'}
    month_names = ['', 'Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
    month_abbreviations = ['', 'Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz', 'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara']
    day_names = ['', 'Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
    day_abbreviations = ['', 'Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz']