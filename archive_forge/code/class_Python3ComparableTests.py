import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
class Python3ComparableTests(SynchronousTestCase):
    """
    Python 3-specific functionality of C{comparable}.
    """

    def test_notImplementedEquals(self):
        """
        Instances of a class that is decorated by C{comparable} support
        returning C{NotImplemented} from C{__eq__} if it is returned by the
        underlying C{__cmp__} call.
        """
        self.assertEqual(Comparable(1).__eq__(object()), NotImplemented)

    def test_notImplementedNotEquals(self):
        """
        Instances of a class that is decorated by C{comparable} support
        returning C{NotImplemented} from C{__ne__} if it is returned by the
        underlying C{__cmp__} call.
        """
        self.assertEqual(Comparable(1).__ne__(object()), NotImplemented)

    def test_notImplementedGreaterThan(self):
        """
        Instances of a class that is decorated by C{comparable} support
        returning C{NotImplemented} from C{__gt__} if it is returned by the
        underlying C{__cmp__} call.
        """
        self.assertEqual(Comparable(1).__gt__(object()), NotImplemented)

    def test_notImplementedLessThan(self):
        """
        Instances of a class that is decorated by C{comparable} support
        returning C{NotImplemented} from C{__lt__} if it is returned by the
        underlying C{__cmp__} call.
        """
        self.assertEqual(Comparable(1).__lt__(object()), NotImplemented)

    def test_notImplementedGreaterThanEquals(self):
        """
        Instances of a class that is decorated by C{comparable} support
        returning C{NotImplemented} from C{__ge__} if it is returned by the
        underlying C{__cmp__} call.
        """
        self.assertEqual(Comparable(1).__ge__(object()), NotImplemented)

    def test_notImplementedLessThanEquals(self):
        """
        Instances of a class that is decorated by C{comparable} support
        returning C{NotImplemented} from C{__le__} if it is returned by the
        underlying C{__cmp__} call.
        """
        self.assertEqual(Comparable(1).__le__(object()), NotImplemented)