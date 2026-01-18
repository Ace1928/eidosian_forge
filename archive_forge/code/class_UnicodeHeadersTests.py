from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
class UnicodeHeadersTests(TestCase):
    """
    Tests for L{Headers}, using L{str} arguments for methods.
    """

    def test_sanitizeLinearWhitespace(self) -> None:
        """
        Linear whitespace in header names or values is replaced with a
        single space.
        """
        assertSanitized(self, textLinearWhitespaceComponents, sanitizedBytes)

    def test_initializer(self) -> None:
        """
        The header values passed to L{Headers.__init__} can be retrieved via
        L{Headers.getRawHeaders}. If a L{bytes} argument is given, it returns
        L{bytes} values, and if a L{str} argument is given, it returns
        L{str} values. Both are the same header value, just encoded or
        decoded.
        """
        h = Headers({'Foo': ['bar']})
        self.assertEqual(h.getRawHeaders(b'foo'), [b'bar'])
        self.assertEqual(h.getRawHeaders('foo'), ['bar'])

    def test_setRawHeaders(self) -> None:
        """
        L{Headers.setRawHeaders} sets the header values for the given
        header name to the sequence of strings, encoded.
        """
        rawValue = ['value1', 'value2']
        rawEncodedValue = [b'value1', b'value2']
        h = Headers()
        h.setRawHeaders('test', rawValue)
        self.assertTrue(h.hasHeader(b'test'))
        self.assertTrue(h.hasHeader(b'Test'))
        self.assertTrue(h.hasHeader('test'))
        self.assertTrue(h.hasHeader('Test'))
        self.assertEqual(h.getRawHeaders('test'), rawValue)
        self.assertEqual(h.getRawHeaders(b'test'), rawEncodedValue)

    def test_nameNotEncodable(self) -> None:
        """
        Passing L{str} to any function that takes a header name will encode
        said header name as ISO-8859-1, and if it cannot be encoded, it will
        raise a L{UnicodeDecodeError}.
        """
        h = Headers()
        with self.assertRaises(UnicodeEncodeError):
            h.setRawHeaders('☃', ['val'])
        with self.assertRaises(UnicodeEncodeError):
            h.hasHeader('☃')

    def test_nameEncoding(self) -> None:
        """
        Passing L{str} to any function that takes a header name will encode
        said header name as ISO-8859-1.
        """
        h = Headers()
        h.setRawHeaders('á', [b'foo'])
        self.assertTrue(h.hasHeader(b'\xe1'))
        self.assertEqual(h.getRawHeaders(b'\xe1'), [b'foo'])
        self.assertTrue(h.hasHeader('á'))

    def test_rawHeadersValueEncoding(self) -> None:
        """
        Passing L{str} to L{Headers.setRawHeaders} will encode the name as
        ISO-8859-1 and values as UTF-8.
        """
        h = Headers()
        h.setRawHeaders('á', ['☃', b'foo'])
        self.assertTrue(h.hasHeader(b'\xe1'))
        self.assertEqual(h.getRawHeaders(b'\xe1'), [b'\xe2\x98\x83', b'foo'])

    def test_rawHeadersTypeChecking(self) -> None:
        """
        L{Headers.setRawHeaders} requires values to be of type sequence
        """
        h = Headers()
        self.assertRaises(TypeError, h.setRawHeaders, 'key', {'Foo': 'bar'})

    def test_addRawHeader(self) -> None:
        """
        L{Headers.addRawHeader} adds a new value for a given header.
        """
        h = Headers()
        h.addRawHeader('test', 'lemur')
        self.assertEqual(h.getRawHeaders('test'), ['lemur'])
        h.addRawHeader('test', 'panda')
        self.assertEqual(h.getRawHeaders('test'), ['lemur', 'panda'])
        self.assertEqual(h.getRawHeaders(b'test'), [b'lemur', b'panda'])

    def test_getRawHeadersNoDefault(self) -> None:
        """
        L{Headers.getRawHeaders} returns L{None} if the header is not found and
        no default is specified.
        """
        self.assertIsNone(Headers().getRawHeaders('test'))

    def test_getRawHeadersDefaultValue(self) -> None:
        """
        L{Headers.getRawHeaders} returns the specified default value when no
        header is found.
        """
        h = Headers()
        default = object()
        self.assertIdentical(h.getRawHeaders('test', default), default)
        self.assertIdentical(h.getRawHeaders('test', None), None)
        self.assertEqual(h.getRawHeaders('test', [None]), [None])
        self.assertEqual(h.getRawHeaders('test', ['☃']), ['☃'])

    def test_getRawHeadersWithDefaultMatchingValue(self) -> None:
        """
        If the object passed as the value list to L{Headers.setRawHeaders}
        is later passed as a default to L{Headers.getRawHeaders}, the
        result nevertheless contains decoded values.
        """
        h = Headers()
        default = [b'value']
        h.setRawHeaders(b'key', default)
        self.assertIsInstance(h.getRawHeaders('key', default)[0], str)
        self.assertEqual(h.getRawHeaders('key', default), ['value'])

    def test_getRawHeaders(self) -> None:
        """
        L{Headers.getRawHeaders} returns the values which have been set for a
        given header.
        """
        h = Headers()
        h.setRawHeaders('testá', ['lemur'])
        self.assertEqual(h.getRawHeaders('testá'), ['lemur'])
        self.assertEqual(h.getRawHeaders('Testá'), ['lemur'])
        self.assertEqual(h.getRawHeaders(b'test\xe1'), [b'lemur'])
        self.assertEqual(h.getRawHeaders(b'Test\xe1'), [b'lemur'])

    def test_hasHeaderTrue(self) -> None:
        """
        Check that L{Headers.hasHeader} returns C{True} when the given header
        is found.
        """
        h = Headers()
        h.setRawHeaders('testá', ['lemur'])
        self.assertTrue(h.hasHeader('testá'))
        self.assertTrue(h.hasHeader('Testá'))
        self.assertTrue(h.hasHeader(b'test\xe1'))
        self.assertTrue(h.hasHeader(b'Test\xe1'))

    def test_hasHeaderFalse(self) -> None:
        """
        L{Headers.hasHeader} returns C{False} when the given header is not
        found.
        """
        self.assertFalse(Headers().hasHeader('testá'))

    def test_removeHeader(self) -> None:
        """
        Check that L{Headers.removeHeader} removes the given header.
        """
        h = Headers()
        h.setRawHeaders('foo', ['lemur'])
        self.assertTrue(h.hasHeader('foo'))
        h.removeHeader('foo')
        self.assertFalse(h.hasHeader('foo'))
        self.assertFalse(h.hasHeader(b'foo'))
        h.setRawHeaders('bar', ['panda'])
        self.assertTrue(h.hasHeader('bar'))
        h.removeHeader('Bar')
        self.assertFalse(h.hasHeader('bar'))
        self.assertFalse(h.hasHeader(b'bar'))

    def test_removeHeaderDoesntExist(self) -> None:
        """
        L{Headers.removeHeader} is a no-operation when the specified header is
        not found.
        """
        h = Headers()
        h.removeHeader('test')
        self.assertEqual(list(h.getAllRawHeaders()), [])

    def test_getAllRawHeaders(self) -> None:
        """
        L{Headers.getAllRawHeaders} returns an iterable of (k, v) pairs, where
        C{k} is the canonicalized representation of the header name, and C{v}
        is a sequence of values.
        """
        h = Headers()
        h.setRawHeaders('testá', ['lemurs'])
        h.setRawHeaders('www-authenticate', ['basic aksljdlk='])
        h.setRawHeaders('content-md5', ['kjdfdfgdfgnsd'])
        allHeaders = {(k, tuple(v)) for k, v in h.getAllRawHeaders()}
        self.assertEqual(allHeaders, {(b'WWW-Authenticate', (b'basic aksljdlk=',)), (b'Content-MD5', (b'kjdfdfgdfgnsd',)), (b'Test\xe1', (b'lemurs',))})

    def test_headersComparison(self) -> None:
        """
        A L{Headers} instance compares equal to itself and to another
        L{Headers} instance with the same values.
        """
        first = Headers()
        first.setRawHeaders('fooá', ['panda'])
        second = Headers()
        second.setRawHeaders('fooá', ['panda'])
        third = Headers()
        third.setRawHeaders('fooá', ['lemur', 'panda'])
        self.assertEqual(first, first)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        firstBytes = Headers()
        firstBytes.setRawHeaders(b'foo\xe1', [b'panda'])
        secondBytes = Headers()
        secondBytes.setRawHeaders(b'foo\xe1', [b'panda'])
        thirdBytes = Headers()
        thirdBytes.setRawHeaders(b'foo\xe1', [b'lemur', 'panda'])
        self.assertEqual(first, firstBytes)
        self.assertEqual(second, secondBytes)
        self.assertEqual(third, thirdBytes)

    def test_otherComparison(self) -> None:
        """
        An instance of L{Headers} does not compare equal to other unrelated
        objects.
        """
        h = Headers()
        self.assertNotEqual(h, ())
        self.assertNotEqual(h, object())
        self.assertNotEqual(h, 'foo')

    def test_repr(self) -> None:
        """
        The L{repr} of a L{Headers} instance shows the names and values of all
        the headers it contains. This shows only reprs of bytes values, as
        undecodable headers may cause an exception.
        """
        foo = 'fooá'
        bar = 'bar☃'
        baz = 'baz'
        fooEncoded = "'foo\\xe1'"
        barEncoded = "'bar\\xe2\\x98\\x83'"
        fooEncoded = 'b' + fooEncoded
        barEncoded = 'b' + barEncoded
        self.assertEqual(repr(Headers({foo: [bar, baz]})), 'Headers({{{}: [{}, {!r}]}})'.format(fooEncoded, barEncoded, baz.encode('utf8')))

    def test_subclassRepr(self) -> None:
        """
        The L{repr} of an instance of a subclass of L{Headers} uses the name
        of the subclass instead of the string C{"Headers"}.
        """
        foo = 'fooá'
        bar = 'bar☃'
        baz = 'baz'
        fooEncoded = "b'foo\\xe1'"
        barEncoded = "b'bar\\xe2\\x98\\x83'"

        class FunnyHeaders(Headers):
            pass
        self.assertEqual(repr(FunnyHeaders({foo: [bar, baz]})), 'FunnyHeaders({%s: [%s, %r]})' % (fooEncoded, barEncoded, baz.encode('utf8')))

    def test_copy(self) -> None:
        """
        L{Headers.copy} creates a new independent copy of an existing
        L{Headers} instance, allowing future modifications without impacts
        between the copies.
        """
        h = Headers()
        h.setRawHeaders('testá', ['foo☃'])
        i = h.copy()
        self.assertEqual(i.getRawHeaders('testá'), ['foo☃'])
        self.assertEqual(i.getRawHeaders(b'test\xe1'), [b'foo\xe2\x98\x83'])
        h.addRawHeader('testá', 'bar')
        self.assertEqual(i.getRawHeaders('testá'), ['foo☃'])
        self.assertEqual(i.getRawHeaders(b'test\xe1'), [b'foo\xe2\x98\x83'])
        i.addRawHeader('testá', b'baz')
        self.assertEqual(h.getRawHeaders('testá'), ['foo☃', 'bar'])
        self.assertEqual(h.getRawHeaders(b'test\xe1'), [b'foo\xe2\x98\x83', b'bar'])