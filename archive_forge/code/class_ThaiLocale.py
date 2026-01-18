import sys
from math import trunc
from typing import (
class ThaiLocale(Locale):
    names = ['th', 'th-th']
    past = '{0} ที่ผ่านมา'
    future = 'ในอีก {0}'
    timeframes = {'now': 'ขณะนี้', 'second': 'วินาที', 'seconds': '{0} ไม่กี่วินาที', 'minute': '1 นาที', 'minutes': '{0} นาที', 'hour': '1 ชั่วโมง', 'hours': '{0} ชั่วโมง', 'day': '1 วัน', 'days': '{0} วัน', 'month': '1 เดือน', 'months': '{0} เดือน', 'year': '1 ปี', 'years': '{0} ปี'}
    month_names = ['', 'มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน', 'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม']
    month_abbreviations = ['', 'ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.', 'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.']
    day_names = ['', 'จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์']
    day_abbreviations = ['', 'จ', 'อ', 'พ', 'พฤ', 'ศ', 'ส', 'อา']
    meridians = {'am': 'am', 'pm': 'pm', 'AM': 'AM', 'PM': 'PM'}
    BE_OFFSET = 543

    def year_full(self, year: int) -> str:
        """Thai always use Buddhist Era (BE) which is CE + 543"""
        year += self.BE_OFFSET
        return f'{year:04d}'

    def year_abbreviation(self, year: int) -> str:
        """Thai always use Buddhist Era (BE) which is CE + 543"""
        year += self.BE_OFFSET
        return f'{year:04d}'[2:]

    def _format_relative(self, humanized: str, timeframe: TimeFrameLiteral, delta: Union[float, int]) -> str:
        """Thai normally doesn't have any space between words"""
        if timeframe == 'now':
            return humanized
        direction = self.past if delta < 0 else self.future
        relative_string = direction.format(humanized)
        if timeframe == 'seconds':
            relative_string = relative_string.replace(' ', '')
        return relative_string