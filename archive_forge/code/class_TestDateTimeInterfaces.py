import unittest
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import tzinfo
from zope.interface.common.idatetime import IDate
from zope.interface.common.idatetime import IDateClass
from zope.interface.common.idatetime import IDateTime
from zope.interface.common.idatetime import IDateTimeClass
from zope.interface.common.idatetime import ITime
from zope.interface.common.idatetime import ITimeClass
from zope.interface.common.idatetime import ITimeDelta
from zope.interface.common.idatetime import ITimeDeltaClass
from zope.interface.common.idatetime import ITZInfo
from zope.interface.verify import verifyClass
from zope.interface.verify import verifyObject
class TestDateTimeInterfaces(unittest.TestCase):

    def test_interfaces(self):
        verifyObject(ITimeDelta, timedelta(minutes=20))
        verifyObject(IDate, date(2000, 1, 2))
        verifyObject(IDateTime, datetime(2000, 1, 2, 10, 20))
        verifyObject(ITime, time(20, 30, 15, 1234))
        verifyObject(ITZInfo, tzinfo())
        verifyClass(ITimeDeltaClass, timedelta)
        verifyClass(IDateClass, date)
        verifyClass(IDateTimeClass, datetime)
        verifyClass(ITimeClass, time)