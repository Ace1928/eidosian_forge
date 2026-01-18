from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
class TestAssertions(TestCase):
    """Test assertions in TestCase."""
    run_test_with = FullStackRunTest

    def raiseError(self, exceptionFactory, *args, **kwargs):
        raise exceptionFactory(*args, **kwargs)

    def test_formatTypes_single(self):

        class Foo:
            pass
        self.assertEqual('Foo', self._formatTypes(Foo))

    def test_formatTypes_multiple(self):

        class Foo:
            pass

        class Bar:
            pass
        self.assertEqual('Foo, Bar', self._formatTypes([Foo, Bar]))

    def test_assertRaises(self):
        self.assertRaises(RuntimeError, self.raiseError, RuntimeError)

    def test_assertRaises_exception_w_metaclass(self):

        class MyExMeta(type):

            def __init__(cls, name, bases, dct):
                """ Do some dummy metaclass stuff """
                dct.update({'answer': 42})
                type.__init__(cls, name, bases, dct)

        class MyEx(Exception):
            __metaclass__ = MyExMeta
        self.assertRaises(MyEx, self.raiseError, MyEx)

    def test_assertRaises_fails_when_no_error_raised(self):
        ret = ('orange', 42)
        self.assertFails("<function ...<lambda> at ...> returned ('orange', 42)", self.assertRaises, RuntimeError, lambda: ret)

    def test_assertRaises_fails_when_different_error_raised(self):
        self.assertThat(lambda: self.assertRaises(RuntimeError, self.raiseError, ZeroDivisionError), Raises(MatchesException(ZeroDivisionError)))

    def test_assertRaises_returns_the_raised_exception(self):
        raisedExceptions = []

        def raiseError():
            try:
                raise RuntimeError('Deliberate error')
            except RuntimeError:
                raisedExceptions.append(sys.exc_info()[1])
                raise
        exception = self.assertRaises(RuntimeError, raiseError)
        self.assertEqual(1, len(raisedExceptions))
        self.assertIs(exception, raisedExceptions[0], '{!r} is not {!r}'.format(exception, raisedExceptions[0]))

    def test_assertRaises_with_multiple_exceptions(self):
        expectedExceptions = (RuntimeError, ZeroDivisionError)
        self.assertRaises(expectedExceptions, self.raiseError, expectedExceptions[0])
        self.assertRaises(expectedExceptions, self.raiseError, expectedExceptions[1])

    def test_assertRaises_with_multiple_exceptions_failure_mode(self):
        expectedExceptions = (RuntimeError, ZeroDivisionError)
        self.assertRaises(self.failureException, self.assertRaises, expectedExceptions, lambda: None)
        self.assertFails('<function ...<lambda> at ...> returned None', self.assertRaises, expectedExceptions, lambda: None)

    def test_assertRaises_function_repr_in_exception(self):

        def foo():
            """An arbitrary function."""
            pass
        self.assertThat(lambda: self.assertRaises(Exception, foo), Raises(MatchesException(self.failureException, f'.*{foo!r}.*')))

    def test_assertRaisesRegex(self):
        self.assertRaisesRegex(RuntimeError, 'M\\w*e', self.raiseError, RuntimeError, 'Message')

    def test_assertRaisesRegex_wrong_error_type(self):
        self.assertRaises(ValueError, self.assertRaisesRegex, RuntimeError, 'M\\w*e', self.raiseError, ValueError, 'Message')

    def test_assertRaisesRegex_wrong_message(self):
        self.assertFails('"Expected" does not match "Observed"', self.assertRaisesRegex, RuntimeError, 'Expected', self.raiseError, RuntimeError, 'Observed')

    def assertFails(self, message, function, *args, **kwargs):
        """Assert that function raises a failure with the given message."""
        failure = self.assertRaises(self.failureException, function, *args, **kwargs)
        self.assertThat(failure, DocTestMatches(message, ELLIPSIS))

    def test_assertIn_success(self):
        self.assertIn(3, range(10))
        self.assertIn('foo', 'foo bar baz')
        self.assertIn('foo', 'foo bar baz'.split())

    def test_assertIn_failure(self):
        self.assertFails('3 not in [0, 1, 2]', self.assertIn, 3, [0, 1, 2])
        self.assertFails('{!r} not in {!r}'.format('qux', 'foo bar baz'), self.assertIn, 'qux', 'foo bar baz')

    def test_assertIn_failure_with_message(self):
        self.assertFails('3 not in [0, 1, 2]: foo bar', self.assertIn, 3, [0, 1, 2], 'foo bar')
        self.assertFails('{!r} not in {!r}: foo bar'.format('qux', 'foo bar baz'), self.assertIn, 'qux', 'foo bar baz', 'foo bar')

    def test_assertNotIn_success(self):
        self.assertNotIn(3, [0, 1, 2])
        self.assertNotIn('qux', 'foo bar baz')

    def test_assertNotIn_failure(self):
        self.assertFails('[1, 2, 3] matches Contains(3)', self.assertNotIn, 3, [1, 2, 3])
        self.assertFails("'foo bar baz' matches Contains('foo')", self.assertNotIn, 'foo', 'foo bar baz')

    def test_assertNotIn_failure_with_message(self):
        self.assertFails('[1, 2, 3] matches Contains(3): foo bar', self.assertNotIn, 3, [1, 2, 3], 'foo bar')
        self.assertFails("'foo bar baz' matches Contains('foo'): foo bar", self.assertNotIn, 'foo', 'foo bar baz', 'foo bar')

    def test_assertIsInstance(self):

        class Foo:
            """Simple class for testing assertIsInstance."""
        foo = Foo()
        self.assertIsInstance(foo, Foo)

    def test_assertIsInstance_multiple_classes(self):

        class Foo:
            """Simple class for testing assertIsInstance."""

        class Bar:
            """Another simple class for testing assertIsInstance."""
        foo = Foo()
        self.assertIsInstance(foo, (Foo, Bar))
        self.assertIsInstance(Bar(), (Foo, Bar))

    def test_assertIsInstance_failure(self):

        class Foo:
            """Simple class for testing assertIsInstance."""
        self.assertFails("'42' is not an instance of %s" % self._formatTypes(Foo), self.assertIsInstance, 42, Foo)

    def test_assertIsInstance_failure_multiple_classes(self):

        class Foo:
            """Simple class for testing assertIsInstance."""

        class Bar:
            """Another simple class for testing assertIsInstance."""
        self.assertFails("'42' is not an instance of any of (%s)" % self._formatTypes([Foo, Bar]), self.assertIsInstance, 42, (Foo, Bar))

    def test_assertIsInstance_overridden_message(self):
        self.assertFails("'42' is not an instance of str: foo", self.assertIsInstance, 42, str, 'foo')

    def test_assertIs(self):
        self.assertIsNone(None)
        some_list = [42]
        self.assertIs(some_list, some_list)
        some_object = object()
        self.assertIs(some_object, some_object)

    def test_assertIs_fails(self):
        self.assertFails('42 is not None', self.assertIs, None, 42)
        self.assertFails('[42] is not [42]', self.assertIs, [42], [42])

    def test_assertIs_fails_with_message(self):
        self.assertFails('42 is not None: foo bar', self.assertIs, None, 42, 'foo bar')

    def test_assertIsNot(self):
        self.assertIsNot(None, 42)
        self.assertIsNot([42], [42])
        self.assertIsNot(object(), object())

    def test_assertIsNot_fails(self):
        self.assertFails('None matches Is(None)', self.assertIsNot, None, None)
        some_list = [42]
        self.assertFails('[42] matches Is([42])', self.assertIsNot, some_list, some_list)

    def test_assertIsNot_fails_with_message(self):
        self.assertFails('None matches Is(None): foo bar', self.assertIsNot, None, None, 'foo bar')

    def test_assertThat_matches_clean(self):

        class Matcher:

            def match(self, foo):
                return None
        self.assertThat('foo', Matcher())

    def test_assertThat_mismatch_raises_description(self):
        calls = []

        class Mismatch:

            def __init__(self, thing):
                self.thing = thing

            def describe(self):
                calls.append(('describe_diff', self.thing))
                return 'object is not a thing'

            def get_details(self):
                return {}

        class Matcher:

            def match(self, thing):
                calls.append(('match', thing))
                return Mismatch(thing)

            def __str__(self):
                calls.append(('__str__',))
                return 'a description'

        class Test(TestCase):

            def test(self):
                self.assertThat('foo', Matcher())
        result = Test('test').run()
        self.assertEqual([('match', 'foo'), ('describe_diff', 'foo')], calls)
        self.assertFalse(result.wasSuccessful())

    def test_assertThat_output(self):
        matchee = 'foo'
        matcher = Equals('bar')
        expected = matcher.match(matchee).describe()
        self.assertFails(expected, self.assertThat, matchee, matcher)

    def test_assertThat_message_is_annotated(self):
        matchee = 'foo'
        matcher = Equals('bar')
        expected = Annotate('woo', matcher).match(matchee).describe()
        self.assertFails(expected, self.assertThat, matchee, matcher, 'woo')

    def test_assertThat_verbose_output(self):
        matchee = 'foo'
        matcher = Equals('bar')
        expected = 'Match failed. Matchee: %r\nMatcher: %s\nDifference: %s\n' % (matchee, matcher, matcher.match(matchee).describe())
        self.assertFails(expected, self.assertThat, matchee, matcher, verbose=True)

    def test_expectThat_matches_clean(self):

        class Matcher:

            def match(self, foo):
                return None
        self.expectThat('foo', Matcher())

    def test_expectThat_mismatch_fails_test(self):

        class Test(TestCase):

            def test(self):
                self.expectThat('foo', Equals('bar'))
        result = Test('test').run()
        self.assertFalse(result.wasSuccessful())

    def test_expectThat_does_not_exit_test(self):

        class Test(TestCase):
            marker = False

            def test(self):
                self.expectThat('foo', Equals('bar'))
                Test.marker = True
        result = Test('test').run()
        self.assertFalse(result.wasSuccessful())
        self.assertTrue(Test.marker)

    def test_expectThat_adds_detail(self):

        class Test(TestCase):

            def test(self):
                self.expectThat('foo', Equals('bar'))
        test = Test('test')
        test.run()
        details = test.getDetails()
        self.assertIn('Failed expectation', details)

    def test__force_failure_fails_test(self):

        class Test(TestCase):

            def test_foo(self):
                self.force_failure = True
                self.remaining_code_run = True
        test = Test('test_foo')
        result = test.run()
        self.assertFalse(result.wasSuccessful())
        self.assertTrue(test.remaining_code_run)

    def get_error_string(self, e):
        """Get the string showing how 'e' would be formatted in test output.

        This is a little bit hacky, since it's designed to give consistent
        output regardless of Python version.

        In testtools, TestResult._exc_info_to_unicode is the point of dispatch
        between various different implementations of methods that format
        exceptions, so that's what we have to call. However, that method cares
        about stack traces and formats the exception class. We don't care
        about either of these, so we take its output and parse it a little.
        """
        error = TracebackContent((e.__class__, e, None), self).as_text()
        if error.startswith('Traceback (most recent call last):\n'):
            lines = error.splitlines(True)[1:]
            for i, line in enumerate(lines):
                if not line.startswith(' '):
                    break
            error = ''.join(lines[i:])
        exc_class, error = error.split(': ', 1)
        return error

    def test_assertThat_verbose_unicode(self):
        matchee = '§'
        matcher = Equals('a')
        expected = 'Match failed. Matchee: %s\nMatcher: %s\nDifference: %s\n\n' % (repr(matchee).replace('\\xa7', matchee), matcher, matcher.match(matchee).describe())
        e = self.assertRaises(self.failureException, self.assertThat, matchee, matcher, verbose=True)
        self.assertEqual(expected, self.get_error_string(e))

    def test_assertEqual_nice_formatting(self):
        message = 'These things ought not be equal.'
        a = ['apple', 'banana', 'cherry']
        b = {'Thatcher': 'One who mends roofs of straw', 'Major': 'A military officer, ranked below colonel', 'Blair': 'To shout loudly', 'Brown': 'The colour of healthy human faeces'}
        expected_error = '\n'.join(['!=:', 'reference = %s' % pformat(a), 'actual    = %s' % pformat(b), ': ' + message])
        self.assertFails(expected_error, self.assertEqual, a, b, message)
        self.assertFails(expected_error, self.assertEquals, a, b, message)
        self.assertFails(expected_error, self.failUnlessEqual, a, b, message)

    def test_assertEqual_formatting_no_message(self):
        a = 'cat'
        b = 'dog'
        expected_error = "'cat' != 'dog'"
        self.assertFails(expected_error, self.assertEqual, a, b)
        self.assertFails(expected_error, self.assertEquals, a, b)
        self.assertFails(expected_error, self.failUnlessEqual, a, b)

    def test_assertEqual_non_ascii_str_with_newlines(self):
        message = 'Be careful mixing unicode and bytes'
        a = 'a\n§\n'
        b = 'Just a longish string so the more verbose output form is used.'
        expected_error = '\n'.join(['!=:', "reference = '''\\", 'a', repr('§')[1:-1], "'''", f'actual    = {b!r}', ': ' + message])
        self.assertFails(expected_error, self.assertEqual, a, b, message)

    def test_assertIsNone(self):
        self.assertIsNone(None)
        expected_error = '0 is not None'
        self.assertFails(expected_error, self.assertIsNone, 0)

    def test_assertIsNotNone(self):
        self.assertIsNotNone(0)
        self.assertIsNotNone('0')
        expected_error = 'None matches Is(None)'
        self.assertFails(expected_error, self.assertIsNotNone, None)

    def test_fail_preserves_traceback_detail(self):

        class Test(TestCase):

            def test(self):
                self.addDetail('traceback', text_content('foo'))
                self.fail('bar')
        test = Test('test')
        result = ExtendedTestResult()
        test.run(result)
        self.assertEqual({'traceback', 'traceback-1'}, set(result._events[1][2].keys()))