import time as _time
import math as _math
import sys
from operator import index as _index
class IsoCalendarDate(tuple):

    def __new__(cls, year, week, weekday, /):
        return super().__new__(cls, (year, week, weekday))

    @property
    def year(self):
        return self[0]

    @property
    def week(self):
        return self[1]

    @property
    def weekday(self):
        return self[2]

    def __reduce__(self):
        return (tuple, (tuple(self),))

    def __repr__(self):
        return f'{self.__class__.__name__}(year={self[0]}, week={self[1]}, weekday={self[2]})'