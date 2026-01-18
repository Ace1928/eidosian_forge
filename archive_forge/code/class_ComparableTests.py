import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
class ComparableTests(SynchronousTestCase):
    """
    L{comparable} decorated classes emulate Python 2's C{__cmp__} semantics.
    """

    def test_equality(self):
        """
        Instances of a class that is decorated by C{comparable} support
        equality comparisons.
        """
        self.assertTrue(Comparable(1) == Comparable(1))
        self.assertFalse(Comparable(2) == Comparable(1))

    def test_nonEquality(self):
        """
        Instances of a class that is decorated by C{comparable} support
        inequality comparisons.
        """
        self.assertFalse(Comparable(1) != Comparable(1))
        self.assertTrue(Comparable(2) != Comparable(1))

    def test_greaterThan(self):
        """
        Instances of a class that is decorated by C{comparable} support
        greater-than comparisons.
        """
        self.assertTrue(Comparable(2) > Comparable(1))
        self.assertFalse(Comparable(0) > Comparable(3))

    def test_greaterThanOrEqual(self):
        """
        Instances of a class that is decorated by C{comparable} support
        greater-than-or-equal comparisons.
        """
        self.assertTrue(Comparable(1) >= Comparable(1))
        self.assertTrue(Comparable(2) >= Comparable(1))
        self.assertFalse(Comparable(0) >= Comparable(3))

    def test_lessThan(self):
        """
        Instances of a class that is decorated by C{comparable} support
        less-than comparisons.
        """
        self.assertTrue(Comparable(0) < Comparable(3))
        self.assertFalse(Comparable(2) < Comparable(0))

    def test_lessThanOrEqual(self):
        """
        Instances of a class that is decorated by C{comparable} support
        less-than-or-equal comparisons.
        """
        self.assertTrue(Comparable(3) <= Comparable(3))
        self.assertTrue(Comparable(0) <= Comparable(3))
        self.assertFalse(Comparable(2) <= Comparable(0))