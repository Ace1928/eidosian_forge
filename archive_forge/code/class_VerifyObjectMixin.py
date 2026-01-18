import unittest
from zope.interface.common import ABCInterface
from zope.interface.common import ABCInterfaceClass
from zope.interface.verify import verifyClass
from zope.interface.verify import verifyObject
class VerifyObjectMixin(VerifyClassMixin):
    verifier = staticmethod(verifyObject)
    CONSTRUCTORS = {}

    def _adjust_object_before_verify(self, iface, x):
        constructor = self.CONSTRUCTORS.get(x)
        if not constructor:
            constructor = self.CONSTRUCTORS.get(iface)
        if not constructor:
            constructor = self.CONSTRUCTORS.get(x.__name__)
        if not constructor:
            constructor = x
        if constructor is unittest.SkipTest:
            self.skipTest('Cannot create ' + str(x))
        result = constructor()
        if hasattr(result, 'close'):
            self.addCleanup(result.close)
        return result