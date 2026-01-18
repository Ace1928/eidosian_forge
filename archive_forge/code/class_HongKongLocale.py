import sys
from math import trunc
from typing import (
class HongKongLocale(Locale):
    names = ['zh-hk']
    past = '{0}前'
    future = '{0}後'
    timeframes = {'now': '剛才', 'second': '1秒', 'seconds': '{0}秒', 'minute': '1分鐘', 'minutes': '{0}分鐘', 'hour': '1小時', 'hours': '{0}小時', 'day': '1天', 'days': '{0}天', 'week': '1星期', 'weeks': '{0}星期', 'month': '1個月', 'months': '{0}個月', 'year': '1年', 'years': '{0}年'}
    month_names = ['', '1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
    month_abbreviations = ['', ' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', '10', '11', '12']
    day_names = ['', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
    day_abbreviations = ['', '一', '二', '三', '四', '五', '六', '日']