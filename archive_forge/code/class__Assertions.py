import inspect
import os
import sys
import tempfile
import types
import unittest as pyunit
import warnings
from dis import findlinestarts as _findlinestarts
from typing import (
from unittest import SkipTest
from attrs import frozen
from typing_extensions import ParamSpec
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python import failure, log, monkey
from twisted.python.deprecate import (
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import itrial, util
class _Assertions(pyunit.TestCase):
    """
    Replaces many of the built-in TestCase assertions. In general, these
    assertions provide better error messages and are easier to use in
    callbacks.
    """

    def fail(self, msg: Optional[object]=None) -> NoReturn:
        """
        Absolutely fail the test.  Do not pass go, do not collect $200.

        @param msg: the message that will be displayed as the reason for the
        failure
        """
        raise self.failureException(msg)

    def assertFalse(self, condition, msg=None):
        """
        Fail the test if C{condition} evaluates to True.

        @param condition: any object that defines __nonzero__
        """
        super().assertFalse(condition, msg)
        return condition
    assertNot = assertFalse
    failUnlessFalse = assertFalse
    failIf = assertFalse

    def assertTrue(self, condition, msg=None):
        """
        Fail the test if C{condition} evaluates to False.

        @param condition: any object that defines __nonzero__
        """
        super().assertTrue(condition, msg)
        return condition
    assert_ = assertTrue
    failUnlessTrue = assertTrue
    failUnless = assertTrue

    def assertRaises(self, exception, f=None, *args, **kwargs):
        """
        Fail the test unless calling the function C{f} with the given
        C{args} and C{kwargs} raises C{exception}. The failure will report
        the traceback and call stack of the unexpected exception.

        @param exception: exception type that is to be expected
        @param f: the function to call

        @return: If C{f} is L{None}, a context manager which will make an
            assertion about the exception raised from the suite it manages.  If
            C{f} is not L{None}, the exception raised by C{f}.

        @raise self.failureException: Raised if the function call does
            not raise an exception or if it raises an exception of a
            different type.
        """
        context = _AssertRaisesContext(self, exception)
        if f is None:
            return context
        return context._handle(lambda: f(*args, **kwargs))
    failUnlessRaises = assertRaises

    def assertEqual(self, first, second, msg=None):
        """
        Fail the test if C{first} and C{second} are not equal.

        @param msg: A string describing the failure that's included in the
            exception.
        """
        super().assertEqual(first, second, msg)
        return first
    failUnlessEqual = assertEqual
    failUnlessEquals = assertEqual
    assertEquals = assertEqual

    def assertIs(self, first, second, msg=None):
        """
        Fail the test if C{first} is not C{second}.  This is an
        obect-identity-equality test, not an object equality
        (i.e. C{__eq__}) test.

        @param msg: if msg is None, then the failure message will be
        '%r is not %r' % (first, second)
        """
        if first is not second:
            raise self.failureException(msg or f'{first!r} is not {second!r}')
        return first
    failUnlessIdentical = assertIs
    assertIdentical = assertIs

    def assertIsNot(self, first, second, msg=None):
        """
        Fail the test if C{first} is C{second}.  This is an
        obect-identity-equality test, not an object equality
        (i.e. C{__eq__}) test.

        @param msg: if msg is None, then the failure message will be
        '%r is %r' % (first, second)
        """
        if first is second:
            raise self.failureException(msg or f'{first!r} is {second!r}')
        return first
    failIfIdentical = assertIsNot
    assertNotIdentical = assertIsNot

    def assertNotEqual(self, first, second, msg=None):
        """
        Fail the test if C{first} == C{second}.

        @param msg: if msg is None, then the failure message will be
        '%r == %r' % (first, second)
        """
        if not first != second:
            raise self.failureException(msg or f'{first!r} == {second!r}')
        return first
    assertNotEquals = assertNotEqual
    failIfEquals = assertNotEqual
    failIfEqual = assertNotEqual

    def assertIn(self, containee, container, msg=None):
        """
        Fail the test if C{containee} is not found in C{container}.

        @param containee: the value that should be in C{container}
        @param container: a sequence type, or in the case of a mapping type,
                          will follow semantics of 'if key in dict.keys()'
        @param msg: if msg is None, then the failure message will be
                    '%r not in %r' % (first, second)
        """
        if containee not in container:
            raise self.failureException(msg or f'{containee!r} not in {container!r}')
        return containee
    failUnlessIn = assertIn

    def assertNotIn(self, containee, container, msg=None):
        """
        Fail the test if C{containee} is found in C{container}.

        @param containee: the value that should not be in C{container}
        @param container: a sequence type, or in the case of a mapping type,
                          will follow semantics of 'if key in dict.keys()'
        @param msg: if msg is None, then the failure message will be
                    '%r in %r' % (first, second)
        """
        if containee in container:
            raise self.failureException(msg or f'{containee!r} in {container!r}')
        return containee
    failIfIn = assertNotIn

    def assertNotAlmostEqual(self, first, second, places=7, msg=None, delta=None):
        """
        Fail if the two objects are equal as determined by their
        difference rounded to the given number of decimal places
        (default 7) and comparing to zero.

        @note: decimal places (from zero) is usually not the same
               as significant digits (measured from the most
               significant digit).

        @note: included for compatibility with PyUnit test cases
        """
        if round(second - first, places) == 0:
            raise self.failureException(msg or f'{first!r} == {second!r} within {places!r} places')
        return first
    assertNotAlmostEquals = assertNotAlmostEqual
    failIfAlmostEqual = assertNotAlmostEqual
    failIfAlmostEquals = assertNotAlmostEqual

    def assertAlmostEqual(self, first, second, places=7, msg=None, delta=None):
        """
        Fail if the two objects are unequal as determined by their
        difference rounded to the given number of decimal places
        (default 7) and comparing to zero.

        @note: decimal places (from zero) is usually not the same
               as significant digits (measured from the most
               significant digit).

        @note: included for compatibility with PyUnit test cases
        """
        if round(second - first, places) != 0:
            raise self.failureException(msg or f'{first!r} != {second!r} within {places!r} places')
        return first
    assertAlmostEquals = assertAlmostEqual
    failUnlessAlmostEqual = assertAlmostEqual

    def assertApproximates(self, first, second, tolerance, msg=None):
        """
        Fail if C{first} - C{second} > C{tolerance}

        @param msg: if msg is None, then the failure message will be
                    '%r ~== %r' % (first, second)
        """
        if abs(first - second) > tolerance:
            raise self.failureException(msg or f'{first} ~== {second}')
        return first
    failUnlessApproximates = assertApproximates

    def assertSubstring(self, substring, astring, msg=None):
        """
        Fail if C{substring} does not exist within C{astring}.
        """
        return self.failUnlessIn(substring, astring, msg)
    failUnlessSubstring = assertSubstring

    def assertNotSubstring(self, substring, astring, msg=None):
        """
        Fail if C{astring} contains C{substring}.
        """
        return self.failIfIn(substring, astring, msg)
    failIfSubstring = assertNotSubstring

    def assertWarns(self, category, message, filename, f, *args, **kwargs):
        """
        Fail if the given function doesn't generate the specified warning when
        called. It calls the function, checks the warning, and forwards the
        result of the function if everything is fine.

        @param category: the category of the warning to check.
        @param message: the output message of the warning to check.
        @param filename: the filename where the warning should come from.
        @param f: the function which is supposed to generate the warning.
        @type f: any callable.
        @param args: the arguments to C{f}.
        @param kwargs: the keywords arguments to C{f}.

        @return: the result of the original function C{f}.
        """
        warningsShown = []
        result = _collectWarnings(warningsShown.append, f, *args, **kwargs)
        if not warningsShown:
            self.fail('No warnings emitted')
        first = warningsShown[0]
        for other in warningsShown[1:]:
            if (other.message, other.category) != (first.message, first.category):
                self.fail("Can't handle different warnings")
        self.assertEqual(first.message, message)
        self.assertIdentical(first.category, category)
        self.assertTrue(filename.startswith(first.filename), f'Warning in {first.filename!r}, expected {filename!r}')
        return result
    failUnlessWarns = assertWarns

    def assertIsInstance(self, instance, classOrTuple, message=None):
        """
        Fail if C{instance} is not an instance of the given class or of
        one of the given classes.

        @param instance: the object to test the type (first argument of the
            C{isinstance} call).
        @type instance: any.
        @param classOrTuple: the class or classes to test against (second
            argument of the C{isinstance} call).
        @type classOrTuple: class, type, or tuple.

        @param message: Custom text to include in the exception text if the
            assertion fails.
        """
        if not isinstance(instance, classOrTuple):
            if message is None:
                suffix = ''
            else:
                suffix = ': ' + message
            self.fail(f'{instance!r} is not an instance of {classOrTuple}{suffix}')
    failUnlessIsInstance = assertIsInstance

    def assertNotIsInstance(self, instance, classOrTuple):
        """
        Fail if C{instance} is an instance of the given class or of one of the
        given classes.

        @param instance: the object to test the type (first argument of the
            C{isinstance} call).
        @type instance: any.
        @param classOrTuple: the class or classes to test against (second
            argument of the C{isinstance} call).
        @type classOrTuple: class, type, or tuple.
        """
        if isinstance(instance, classOrTuple):
            self.fail(f'{instance!r} is an instance of {classOrTuple}')
    failIfIsInstance = assertNotIsInstance

    def successResultOf(self, deferred: Union[Coroutine[Deferred[T], Any, T], Generator[Deferred[T], Any, T], Deferred[T]]) -> T:
        """
        Return the current success result of C{deferred} or raise
        C{self.failureException}.

        @param deferred: A L{Deferred<twisted.internet.defer.Deferred>} or
            I{coroutine} which has a success result.

            For a L{Deferred<twisted.internet.defer.Deferred>} this means
            L{Deferred.callback<twisted.internet.defer.Deferred.callback>} or
            L{Deferred.errback<twisted.internet.defer.Deferred.errback>} has
            been called on it and it has reached the end of its callback chain
            and the last callback or errback returned a
            non-L{failure.Failure}.

            For a I{coroutine} this means all awaited values have a success
            result.

        @raise SynchronousTestCase.failureException: If the
            L{Deferred<twisted.internet.defer.Deferred>} has no result or has a
            failure result.

        @return: The result of C{deferred}.
        """
        deferred = ensureDeferred(deferred)
        results: List[Union[T, failure.Failure]] = []
        deferred.addBoth(results.append)
        if not results:
            self.fail('Success result expected on {!r}, found no result instead'.format(deferred))
        result = results[0]
        if isinstance(result, failure.Failure):
            self.fail('Success result expected on {!r}, found failure result instead:\n{}'.format(deferred, result.getTraceback()))
        return result

    def failureResultOf(self, deferred, *expectedExceptionTypes):
        """
        Return the current failure result of C{deferred} or raise
        C{self.failureException}.

        @param deferred: A L{Deferred<twisted.internet.defer.Deferred>} which
            has a failure result.  This means
            L{Deferred.callback<twisted.internet.defer.Deferred.callback>} or
            L{Deferred.errback<twisted.internet.defer.Deferred.errback>} has
            been called on it and it has reached the end of its callback chain
            and the last callback or errback raised an exception or returned a
            L{failure.Failure}.
        @type deferred: L{Deferred<twisted.internet.defer.Deferred>}

        @param expectedExceptionTypes: Exception types to expect - if
            provided, and the exception wrapped by the failure result is
            not one of the types provided, then this test will fail.

        @raise SynchronousTestCase.failureException: If the
            L{Deferred<twisted.internet.defer.Deferred>} has no result, has a
            success result, or has an unexpected failure result.

        @return: The failure result of C{deferred}.
        @rtype: L{failure.Failure}
        """
        deferred = ensureDeferred(deferred)
        result = []
        deferred.addBoth(result.append)
        if not result:
            self.fail('Failure result expected on {!r}, found no result instead'.format(deferred))
        result = result[0]
        if not isinstance(result, failure.Failure):
            self.fail('Failure result expected on {!r}, found success result ({!r}) instead'.format(deferred, result))
        if expectedExceptionTypes and (not result.check(*expectedExceptionTypes)):
            expectedString = ' or '.join(['.'.join((t.__module__, t.__name__)) for t in expectedExceptionTypes])
            self.fail('Failure of type ({}) expected on {!r}, found type {!r} instead: {}'.format(expectedString, deferred, result.type, result.getTraceback()))
        return result

    def assertNoResult(self, deferred):
        """
        Assert that C{deferred} does not have a result at this point.

        If the assertion succeeds, then the result of C{deferred} is left
        unchanged. Otherwise, any L{failure.Failure} result is swallowed.

        @param deferred: A L{Deferred<twisted.internet.defer.Deferred>} without
            a result.  This means that neither
            L{Deferred.callback<twisted.internet.defer.Deferred.callback>} nor
            L{Deferred.errback<twisted.internet.defer.Deferred.errback>} has
            been called, or that the
            L{Deferred<twisted.internet.defer.Deferred>} is waiting on another
            L{Deferred<twisted.internet.defer.Deferred>} for a result.
        @type deferred: L{Deferred<twisted.internet.defer.Deferred>}

        @raise SynchronousTestCase.failureException: If the
            L{Deferred<twisted.internet.defer.Deferred>} has a result.
        """
        deferred = ensureDeferred(deferred)
        result = []

        def cb(res):
            result.append(res)
            return res
        deferred.addBoth(cb)
        if result:
            deferred.addErrback(lambda _: None)
            self.fail('No result expected on {!r}, found {!r} instead'.format(deferred, result[0]))