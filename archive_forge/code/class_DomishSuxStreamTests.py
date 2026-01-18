from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
class DomishSuxStreamTests(DomishStreamTestsMixin, unittest.TestCase):
    """
    Tests for L{domish.SuxElementStream}, the L{twisted.web.sux}-based element
    stream implementation.
    """
    streamClass = domish.SuxElementStream