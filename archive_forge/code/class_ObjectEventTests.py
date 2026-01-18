import unittest
class ObjectEventTests(unittest.TestCase, _ConformsToIObjectEvent):

    def _getTargetClass(self):
        from zope.interface.interfaces import ObjectEvent
        return ObjectEvent

    def test_ctor(self):
        target = object()
        event = self._makeOne(target)
        self.assertTrue(event.object is target)