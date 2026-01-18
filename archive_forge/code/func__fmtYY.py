import re, time, datetime
from .utils import isStr
def _fmtYY(self):
    return '%02d' % (self.year() % 100)