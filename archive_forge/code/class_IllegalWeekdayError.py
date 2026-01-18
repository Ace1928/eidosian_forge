import sys
import datetime
import locale as _locale
from itertools import repeat
class IllegalWeekdayError(ValueError):

    def __init__(self, weekday):
        self.weekday = weekday

    def __str__(self):
        return 'bad weekday number %r; must be 0 (Monday) to 6 (Sunday)' % self.weekday