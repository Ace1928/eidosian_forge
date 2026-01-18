from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
class DomishExpatStreamTests(DomishStreamTestsMixin, unittest.TestCase):
    """
    Tests for L{domish.ExpatElementStream}, the expat-based element stream
    implementation.
    """
    streamClass = domish.ExpatElementStream
    if requireModule('pyexpat', default=None) is None:
        skip = 'pyexpat is required for ExpatElementStream tests.'