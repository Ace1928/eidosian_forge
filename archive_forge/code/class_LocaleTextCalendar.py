import sys
import datetime
import locale as _locale
from itertools import repeat
class LocaleTextCalendar(TextCalendar):
    """
    This class can be passed a locale name in the constructor and will return
    month and weekday names in the specified locale.
    """

    def __init__(self, firstweekday=0, locale=None):
        TextCalendar.__init__(self, firstweekday)
        if locale is None:
            locale = _get_default_locale()
        self.locale = locale

    def formatweekday(self, day, width):
        with different_locale(self.locale):
            return super().formatweekday(day, width)

    def formatmonthname(self, theyear, themonth, width, withyear=True):
        with different_locale(self.locale):
            return super().formatmonthname(theyear, themonth, width, withyear)