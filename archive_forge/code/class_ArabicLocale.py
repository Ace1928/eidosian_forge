import sys
from math import trunc
from typing import (
class ArabicLocale(Locale):
    names = ['ar', 'ar-ae', 'ar-bh', 'ar-dj', 'ar-eg', 'ar-eh', 'ar-er', 'ar-km', 'ar-kw', 'ar-ly', 'ar-om', 'ar-qa', 'ar-sa', 'ar-sd', 'ar-so', 'ar-ss', 'ar-td', 'ar-ye']
    past = 'منذ {0}'
    future = 'خلال {0}'
    timeframes: ClassVar[Mapping[TimeFrameLiteral, Union[str, Mapping[str, str]]]] = {'now': 'الآن', 'second': 'ثانية', 'seconds': {'2': 'ثانيتين', 'ten': '{0} ثوان', 'higher': '{0} ثانية'}, 'minute': 'دقيقة', 'minutes': {'2': 'دقيقتين', 'ten': '{0} دقائق', 'higher': '{0} دقيقة'}, 'hour': 'ساعة', 'hours': {'2': 'ساعتين', 'ten': '{0} ساعات', 'higher': '{0} ساعة'}, 'day': 'يوم', 'days': {'2': 'يومين', 'ten': '{0} أيام', 'higher': '{0} يوم'}, 'week': 'اسبوع', 'weeks': {'2': 'اسبوعين', 'ten': '{0} أسابيع', 'higher': '{0} اسبوع'}, 'month': 'شهر', 'months': {'2': 'شهرين', 'ten': '{0} أشهر', 'higher': '{0} شهر'}, 'year': 'سنة', 'years': {'2': 'سنتين', 'ten': '{0} سنوات', 'higher': '{0} سنة'}}
    month_names = ['', 'يناير', 'فبراير', 'مارس', 'أبريل', 'مايو', 'يونيو', 'يوليو', 'أغسطس', 'سبتمبر', 'أكتوبر', 'نوفمبر', 'ديسمبر']
    month_abbreviations = ['', 'يناير', 'فبراير', 'مارس', 'أبريل', 'مايو', 'يونيو', 'يوليو', 'أغسطس', 'سبتمبر', 'أكتوبر', 'نوفمبر', 'ديسمبر']
    day_names = ['', 'الإثنين', 'الثلاثاء', 'الأربعاء', 'الخميس', 'الجمعة', 'السبت', 'الأحد']
    day_abbreviations = ['', 'إثنين', 'ثلاثاء', 'أربعاء', 'خميس', 'جمعة', 'سبت', 'أحد']

    def _format_timeframe(self, timeframe: TimeFrameLiteral, delta: int) -> str:
        form = self.timeframes[timeframe]
        delta = abs(delta)
        if isinstance(form, Mapping):
            if delta == 2:
                form = form['2']
            elif 2 < delta <= 10:
                form = form['ten']
            else:
                form = form['higher']
        return form.format(delta)