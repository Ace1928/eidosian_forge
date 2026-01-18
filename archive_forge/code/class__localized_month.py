import sys
import datetime
import locale as _locale
from itertools import repeat
class _localized_month:
    _months = [datetime.date(2001, i + 1, 1).strftime for i in range(12)]
    _months.insert(0, lambda x: '')

    def __init__(self, format):
        self.format = format

    def __getitem__(self, i):
        funcs = self._months[i]
        if isinstance(i, slice):
            return [f(self.format) for f in funcs]
        else:
            return funcs(self.format)

    def __len__(self):
        return 13