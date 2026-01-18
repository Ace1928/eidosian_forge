import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
class SafeStrTests(TestCase):
    """
    Tests for L{reflect.safe_str} function.
    """

    def test_workingStr(self):
        x = [1, 2, 3]
        self.assertEqual(reflect.safe_str(x), str(x))

    def test_brokenStr(self):
        b = Breakable()
        b.breakStr = True
        reflect.safe_str(b)

    def test_workingAscii(self):
        """
        L{safe_str} for C{str} with ascii-only data should return the
        value unchanged.
        """
        x = 'a'
        self.assertEqual(reflect.safe_str(x), 'a')

    def test_workingUtf8_3(self):
        """
        L{safe_str} for C{bytes} with utf-8 encoded data should return
        the value decoded into C{str}.
        """
        x = b't\xc3\xbcst'
        self.assertEqual(reflect.safe_str(x), x.decode('utf-8'))

    def test_brokenUtf8(self):
        """
        Use str() for non-utf8 bytes: "b'non-utf8'"
        """
        x = b'\xff'
        xStr = reflect.safe_str(x)
        self.assertEqual(xStr, str(x))

    def test_brokenRepr(self):
        b = Breakable()
        b.breakRepr = True
        reflect.safe_str(b)

    def test_brokenClassStr(self):

        class X(BTBase):
            breakStr = True
        reflect.safe_str(X)
        reflect.safe_str(X())

    def test_brokenClassRepr(self):

        class X(BTBase):
            breakRepr = True
        reflect.safe_str(X)
        reflect.safe_str(X())

    def test_brokenClassAttribute(self):
        """
        If an object raises an exception when accessing its C{__class__}
        attribute, L{reflect.safe_str} uses C{type} to retrieve the class
        object.
        """
        b = NoClassAttr()
        b.breakStr = True
        bStr = reflect.safe_str(b)
        self.assertIn('NoClassAttr instance at 0x', bStr)
        self.assertIn(os.path.splitext(__file__)[0], bStr)
        self.assertIn('RuntimeError: str!', bStr)

    def test_brokenClassNameAttribute(self):
        """
        If a class raises an exception when accessing its C{__name__} attribute
        B{and} when calling its C{__str__} implementation, L{reflect.safe_str}
        returns 'BROKEN CLASS' instead of the class name.
        """

        class X(BTBase):
            breakName = True
        xStr = reflect.safe_str(X())
        self.assertIn('<BROKEN CLASS AT 0x', xStr)
        self.assertIn(os.path.splitext(__file__)[0], xStr)
        self.assertIn('RuntimeError: str!', xStr)