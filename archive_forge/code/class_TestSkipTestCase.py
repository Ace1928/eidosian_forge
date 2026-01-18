from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
@skipIf(True, 'skipping this test')
class TestSkipTestCase(unittest.SynchronousTestCase):
    pass