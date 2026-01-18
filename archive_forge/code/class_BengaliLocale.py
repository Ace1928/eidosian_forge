import sys
from math import trunc
from typing import (
class BengaliLocale(Locale):
    names = ['bn', 'bn-bd', 'bn-in']
    past = '{0} আগে'
    future = '{0} পরে'
    timeframes = {'now': 'এখন', 'second': 'একটি দ্বিতীয়', 'seconds': '{0} সেকেন্ড', 'minute': 'এক মিনিট', 'minutes': '{0} মিনিট', 'hour': 'এক ঘণ্টা', 'hours': '{0} ঘণ্টা', 'day': 'এক দিন', 'days': '{0} দিন', 'month': 'এক মাস', 'months': '{0} মাস ', 'year': 'এক বছর', 'years': '{0} বছর'}
    meridians = {'am': 'সকাল', 'pm': 'বিকাল', 'AM': 'সকাল', 'PM': 'বিকাল'}
    month_names = ['', 'জানুয়ারি', 'ফেব্রুয়ারি', 'মার্চ', 'এপ্রিল', 'মে', 'জুন', 'জুলাই', 'আগস্ট', 'সেপ্টেম্বর', 'অক্টোবর', 'নভেম্বর', 'ডিসেম্বর']
    month_abbreviations = ['', 'জানু', 'ফেব', 'মার্চ', 'এপ্রি', 'মে', 'জুন', 'জুল', 'অগা', 'সেপ্ট', 'অক্টো', 'নভে', 'ডিসে']
    day_names = ['', 'সোমবার', 'মঙ্গলবার', 'বুধবার', 'বৃহস্পতিবার', 'শুক্রবার', 'শনিবার', 'রবিবার']
    day_abbreviations = ['', 'সোম', 'মঙ্গল', 'বুধ', 'বৃহঃ', 'শুক্র', 'শনি', 'রবি']

    def _ordinal_number(self, n: int) -> str:
        if n > 10 or n == 0:
            return f'{n}তম'
        if n in [1, 5, 7, 8, 9, 10]:
            return f'{n}ম'
        if n in [2, 3]:
            return f'{n}য়'
        if n == 4:
            return f'{n}র্থ'
        if n == 6:
            return f'{n}ষ্ঠ'
        return ''