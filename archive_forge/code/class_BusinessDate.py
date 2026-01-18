import re, time, datetime
from .utils import isStr
class BusinessDate(NormalDate):
    """
    Specialised NormalDate
    """

    def add(self, days):
        """add days to date; use negative integers to subtract"""
        if not isinstance(days, int):
            raise NormalDateException('add method parameter must be integer')
        self.normalize(self.scalar() + days)

    def __add__(self, days):
        """add integer to BusinessDate and return a new, calculated value"""
        if not isinstance(days, int):
            raise NormalDateException('__add__ parameter must be integer')
        cloned = self.clone()
        cloned.add(days)
        return cloned

    def __sub__(self, v):
        return isinstance(v, int) and self.__add__(-v) or self.scalar() - v.scalar()

    def asNormalDate(self):
        return ND(self.normalDate)

    def daysBetweenDates(self, normalDate):
        return self.asNormalDate.daysBetweenDates(normalDate)

    def _checkDOW(self):
        if self.dayOfWeek() > 4:
            raise NormalDateException("%r isn't a business day" % self.normalDate)

    def normalize(self, i):
        i = int(i)
        NormalDate.normalize(self, i // 5 * 7 + i % 5 + BDEpochScalar)

    def scalar(self):
        d = self.asNormalDate()
        i = d - BDEpoch
        return 5 * (i // 7) + i % 7

    def setNormalDate(self, normalDate):
        NormalDate.setNormalDate(self, normalDate)
        self._checkDOW()