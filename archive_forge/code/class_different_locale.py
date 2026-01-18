import sys
import datetime
import locale as _locale
from itertools import repeat
class different_locale:

    def __init__(self, locale):
        self.locale = locale
        self.oldlocale = None

    def __enter__(self):
        self.oldlocale = _locale.setlocale(_locale.LC_TIME, None)
        _locale.setlocale(_locale.LC_TIME, self.locale)

    def __exit__(self, *args):
        if self.oldlocale is None:
            return
        _locale.setlocale(_locale.LC_TIME, self.oldlocale)