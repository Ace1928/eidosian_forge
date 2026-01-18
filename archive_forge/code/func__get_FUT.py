import unittest
@classmethod
def _get_FUT(cls):
    from zope.interface.verify import verifyObject
    return verifyObject