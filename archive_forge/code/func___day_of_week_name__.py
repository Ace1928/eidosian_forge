import re, time, datetime
from .utils import isStr
@property
def __day_of_week_name__(self):
    return getattr(self, '_dayOfWeekName', _dayOfWeekName)