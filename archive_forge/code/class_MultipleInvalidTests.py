import unittest
class MultipleInvalidTests(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.exceptions import MultipleInvalid
        return MultipleInvalid

    def _makeOne(self, excs):
        iface = _makeIface()
        return self._getTargetClass()(iface, 'target', excs)

    def test__str__(self):
        from zope.interface.exceptions import BrokenMethodImplementation
        excs = [BrokenMethodImplementation('aMethod', 'I said so'), Exception('Regular exception')]
        dni = self._makeOne(excs)
        self.assertEqual(str(dni), "The object 'target' has failed to implement interface zope.interface.tests.test_exceptions.IDummy:\n    The contract of 'aMethod' is violated because I said so\n    Regular exception")

    def test__repr__(self):
        from zope.interface.exceptions import BrokenMethodImplementation
        excs = [BrokenMethodImplementation('aMethod', 'I said so'), Exception('Regular', 'exception')]
        dni = self._makeOne(excs)
        self.assertEqual(repr(dni), "MultipleInvalid(<InterfaceClass zope.interface.tests.test_exceptions.IDummy>, 'target', (BrokenMethodImplementation('aMethod', 'I said so'), Exception('Regular', 'exception')))")