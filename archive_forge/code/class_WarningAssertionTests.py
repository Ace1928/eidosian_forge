import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
class WarningAssertionTests(unittest.SynchronousTestCase):

    def test_assertWarns(self):
        """
        Test basic assertWarns report.
        """

        def deprecated(a):
            warnings.warn('Woo deprecated', category=DeprecationWarning)
            return a
        r = self.assertWarns(DeprecationWarning, 'Woo deprecated', __file__, deprecated, 123)
        self.assertEqual(r, 123)

    def test_assertWarnsRegistryClean(self):
        """
        Test that assertWarns cleans the warning registry, so the warning is
        not swallowed the second time.
        """

        def deprecated(a):
            warnings.warn('Woo deprecated', category=DeprecationWarning)
            return a
        r1 = self.assertWarns(DeprecationWarning, 'Woo deprecated', __file__, deprecated, 123)
        self.assertEqual(r1, 123)
        r2 = self.assertWarns(DeprecationWarning, 'Woo deprecated', __file__, deprecated, 321)
        self.assertEqual(r2, 321)

    def test_assertWarnsError(self):
        """
        Test assertWarns failure when no warning is generated.
        """

        def normal(a):
            return a
        self.assertRaises(self.failureException, self.assertWarns, DeprecationWarning, 'Woo deprecated', __file__, normal, 123)

    def test_assertWarnsWrongCategory(self):
        """
        Test assertWarns failure when the category is wrong.
        """

        def deprecated(a):
            warnings.warn('Foo deprecated', category=DeprecationWarning)
            return a
        self.assertRaises(self.failureException, self.assertWarns, UserWarning, 'Foo deprecated', __file__, deprecated, 123)

    def test_assertWarnsWrongMessage(self):
        """
        Test assertWarns failure when the message is wrong.
        """

        def deprecated(a):
            warnings.warn('Foo deprecated', category=DeprecationWarning)
            return a
        self.assertRaises(self.failureException, self.assertWarns, DeprecationWarning, 'Bar deprecated', __file__, deprecated, 123)

    def test_assertWarnsWrongFile(self):
        """
        If the warning emitted by a function refers to a different file than is
        passed to C{assertWarns}, C{failureException} is raised.
        """

        def deprecated(a):
            warnings.warn('Foo deprecated', category=DeprecationWarning, stacklevel=2)
        self.assertRaises(self.failureException, self.assertWarns, DeprecationWarning, 'Foo deprecated', __file__, deprecated, 123)

    def test_assertWarnsOnClass(self):
        """
        Test assertWarns works when creating a class instance.
        """

        class Warn:

            def __init__(self):
                warnings.warn('Do not call me', category=RuntimeWarning)
        r = self.assertWarns(RuntimeWarning, 'Do not call me', __file__, Warn)
        self.assertTrue(isinstance(r, Warn))
        r = self.assertWarns(RuntimeWarning, 'Do not call me', __file__, Warn)
        self.assertTrue(isinstance(r, Warn))

    def test_assertWarnsOnMethod(self):
        """
        Test assertWarns works when used on an instance method.
        """

        class Warn:

            def deprecated(self, a):
                warnings.warn('Bar deprecated', category=DeprecationWarning)
                return a
        w = Warn()
        r = self.assertWarns(DeprecationWarning, 'Bar deprecated', __file__, w.deprecated, 321)
        self.assertEqual(r, 321)
        r = self.assertWarns(DeprecationWarning, 'Bar deprecated', __file__, w.deprecated, 321)
        self.assertEqual(r, 321)

    def test_assertWarnsOnCall(self):
        """
        Test assertWarns works on instance with C{__call__} method.
        """

        class Warn:

            def __call__(self, a):
                warnings.warn('Egg deprecated', category=DeprecationWarning)
                return a
        w = Warn()
        r = self.assertWarns(DeprecationWarning, 'Egg deprecated', __file__, w, 321)
        self.assertEqual(r, 321)
        r = self.assertWarns(DeprecationWarning, 'Egg deprecated', __file__, w, 321)
        self.assertEqual(r, 321)

    def test_assertWarnsFilter(self):
        """
        Test assertWarns on a warning filtered by default.
        """

        def deprecated(a):
            warnings.warn('Woo deprecated', category=PendingDeprecationWarning)
            return a
        r = self.assertWarns(PendingDeprecationWarning, 'Woo deprecated', __file__, deprecated, 123)
        self.assertEqual(r, 123)

    def test_assertWarnsMultipleWarnings(self):
        """
        C{assertWarns} does not raise an exception if the function it is passed
        triggers the same warning more than once.
        """

        def deprecated():
            warnings.warn('Woo deprecated', category=PendingDeprecationWarning)

        def f():
            deprecated()
            deprecated()
        self.assertWarns(PendingDeprecationWarning, 'Woo deprecated', __file__, f)

    def test_assertWarnsDifferentWarnings(self):
        """
        For now, assertWarns is unable to handle multiple different warnings,
        so it should raise an exception if it's the case.
        """

        def deprecated(a):
            warnings.warn('Woo deprecated', category=DeprecationWarning)
            warnings.warn('Another one', category=PendingDeprecationWarning)
        e = self.assertRaises(self.failureException, self.assertWarns, DeprecationWarning, 'Woo deprecated', __file__, deprecated, 123)
        self.assertEqual(str(e), "Can't handle different warnings")

    def test_assertWarnsAfterUnassertedWarning(self):
        """
        Warnings emitted before L{TestCase.assertWarns} is called do not get
        flushed and do not alter the behavior of L{TestCase.assertWarns}.
        """

        class TheWarning(Warning):
            pass

        def f(message):
            warnings.warn(message, category=TheWarning)
        f('foo')
        self.assertWarns(TheWarning, 'bar', __file__, f, 'bar')
        [warning] = self.flushWarnings([f])
        self.assertEqual(warning['message'], 'foo')