import sys
from math import trunc
from typing import (
class LaotianLocale(Locale):
    names = ['lo', 'lo-la']
    past = '{0} ກ່ອນຫນ້ານີ້'
    future = 'ໃນ {0}'
    timeframes = {'now': 'ດຽວນີ້', 'second': 'ວິນາທີ', 'seconds': '{0} ວິນາທີ', 'minute': 'ນາທີ', 'minutes': '{0} ນາທີ', 'hour': 'ຊົ່ວໂມງ', 'hours': '{0} ຊົ່ວໂມງ', 'day': 'ມື້', 'days': '{0} ມື້', 'week': 'ອາທິດ', 'weeks': '{0} ອາທິດ', 'month': 'ເດືອນ', 'months': '{0} ເດືອນ', 'year': 'ປີ', 'years': '{0} ປີ'}
    month_names = ['', 'ມັງກອນ', 'ກຸມພາ', 'ມີນາ', 'ເມສາ', 'ພຶດສະພາ', 'ມິຖຸນາ', 'ກໍລະກົດ', 'ສິງຫາ', 'ກັນຍາ', 'ຕຸລາ', 'ພະຈິກ', 'ທັນວາ']
    month_abbreviations = ['', 'ມັງກອນ', 'ກຸມພາ', 'ມີນາ', 'ເມສາ', 'ພຶດສະພາ', 'ມິຖຸນາ', 'ກໍລະກົດ', 'ສິງຫາ', 'ກັນຍາ', 'ຕຸລາ', 'ພະຈິກ', 'ທັນວາ']
    day_names = ['', 'ວັນຈັນ', 'ວັນອັງຄານ', 'ວັນພຸດ', 'ວັນພະຫັດ', 'ວັນ\u200bສຸກ', 'ວັນເສົາ', 'ວັນອາທິດ']
    day_abbreviations = ['', 'ວັນຈັນ', 'ວັນອັງຄານ', 'ວັນພຸດ', 'ວັນພະຫັດ', 'ວັນ\u200bສຸກ', 'ວັນເສົາ', 'ວັນອາທິດ']
    BE_OFFSET = 543

    def year_full(self, year: int) -> str:
        """Lao always use Buddhist Era (BE) which is CE + 543"""
        year += self.BE_OFFSET
        return f'{year:04d}'

    def year_abbreviation(self, year: int) -> str:
        """Lao always use Buddhist Era (BE) which is CE + 543"""
        year += self.BE_OFFSET
        return f'{year:04d}'[2:]

    def _format_relative(self, humanized: str, timeframe: TimeFrameLiteral, delta: Union[float, int]) -> str:
        """Lao normally doesn't have any space between words"""
        if timeframe == 'now':
            return humanized
        direction = self.past if delta < 0 else self.future
        relative_string = direction.format(humanized)
        if timeframe == 'seconds':
            relative_string = relative_string.replace(' ', '')
        return relative_string