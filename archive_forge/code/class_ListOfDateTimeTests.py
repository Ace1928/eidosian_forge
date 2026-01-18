import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
class ListOfDateTimeTests(TestCase, ListOfTestsMixin):
    """
    Tests for L{ListOf} combined with L{amp.DateTime}.
    """
    elementType = amp.DateTime()
    strings = {b'christmas': b'\x00 2010-12-25T00:00:00.000000-00:00\x00 2010-12-25T00:00:00.000000-00:00', b'christmas in eu': b'\x00 2010-12-25T00:00:00.000000+01:00', b'christmas in iran': b'\x00 2010-12-25T00:00:00.000000+03:30', b'christmas in nyc': b'\x00 2010-12-25T00:00:00.000000-05:00', b'previous tests': b'\x00 2010-12-25T00:00:00.000000+03:19\x00 2010-12-25T00:00:00.000000-06:59'}
    objects = {'christmas': [datetime.datetime(2010, 12, 25, 0, 0, 0, tzinfo=amp.utc), datetime.datetime(2010, 12, 25, 0, 0, 0, tzinfo=tz('+', 0, 0))], 'christmas in eu': [datetime.datetime(2010, 12, 25, 0, 0, 0, tzinfo=tz('+', 1, 0))], 'christmas in iran': [datetime.datetime(2010, 12, 25, 0, 0, 0, tzinfo=tz('+', 3, 30))], 'christmas in nyc': [datetime.datetime(2010, 12, 25, 0, 0, 0, tzinfo=tz('-', 5, 0))], 'previous tests': [datetime.datetime(2010, 12, 25, 0, 0, 0, tzinfo=tz('+', 3, 19)), datetime.datetime(2010, 12, 25, 0, 0, 0, tzinfo=tz('-', 6, 59))]}