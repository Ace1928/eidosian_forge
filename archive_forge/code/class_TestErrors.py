from unittest import TestCase
from fastimport import (
class TestErrors(TestCase):

    def test_MissingBytes(self):
        e = errors.MissingBytes(99, 10, 8)
        self.assertEqual('line 99: Unexpected EOF - expected 10 bytes, found 8', str(e))

    def test_MissingTerminator(self):
        e = errors.MissingTerminator(99, '---')
        self.assertEqual("line 99: Unexpected EOF - expected '---' terminator", str(e))

    def test_InvalidCommand(self):
        e = errors.InvalidCommand(99, 'foo')
        self.assertEqual("line 99: Invalid command 'foo'", str(e))

    def test_MissingSection(self):
        e = errors.MissingSection(99, 'foo', 'bar')
        self.assertEqual('line 99: Command foo is missing section bar', str(e))

    def test_BadFormat(self):
        e = errors.BadFormat(99, 'foo', 'bar', 'xyz')
        self.assertEqual("line 99: Bad format for section bar in command foo: found 'xyz'", str(e))

    def test_InvalidTimezone(self):
        e = errors.InvalidTimezone(99, 'aa:bb')
        self.assertEqual('aa:bb', e.timezone)
        self.assertEqual('', e.reason)
        self.assertEqual("line 99: Timezone 'aa:bb' could not be converted.", str(e))
        e = errors.InvalidTimezone(99, 'aa:bb', 'Non-numeric hours')
        self.assertEqual('aa:bb', e.timezone)
        self.assertEqual(' Non-numeric hours', e.reason)
        self.assertEqual("line 99: Timezone 'aa:bb' could not be converted. Non-numeric hours", str(e))

    def test_UnknownDateFormat(self):
        e = errors.UnknownDateFormat('aaa')
        self.assertEqual("Unknown date format 'aaa'", str(e))

    def test_MissingHandler(self):
        e = errors.MissingHandler('foo')
        self.assertEqual('Missing handler for command foo', str(e))

    def test_UnknownFeature(self):
        e = errors.UnknownFeature('aaa')
        self.assertEqual("Unknown feature 'aaa' - try a later importer or an earlier data format", str(e))