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
class TestDetailsProvided(TestWithDetails):
    run_test_with = FullStackRunTest

    def test_addDetail(self):
        mycontent = self.get_content()
        self.addDetail('foo', mycontent)
        details = self.getDetails()
        self.assertEqual({'foo': mycontent}, details)

    def test_addError(self):

        class Case(TestCase):

            def test(this):
                this.addDetail('foo', self.get_content())
                1 / 0
        self.assertDetailsProvided(Case('test'), 'addError', ['foo', 'traceback'])

    def test_addFailure(self):

        class Case(TestCase):

            def test(this):
                this.addDetail('foo', self.get_content())
                self.fail('yo')
        self.assertDetailsProvided(Case('test'), 'addFailure', ['foo', 'traceback'])

    def test_addSkip(self):

        class Case(TestCase):

            def test(this):
                this.addDetail('foo', self.get_content())
                self.skipTest('yo')
        self.assertDetailsProvided(Case('test'), 'addSkip', ['foo', 'reason'])

    def test_addSkip_different_exception(self):

        class Case(TestCase):
            skipException = ValueError

            def test(this):
                this.addDetail('foo', self.get_content())
                this.skipTest('yo')
        self.assertDetailsProvided(Case('test'), 'addSkip', ['foo', 'reason'])

    def test_addSucccess(self):

        class Case(TestCase):

            def test(this):
                this.addDetail('foo', self.get_content())
        self.assertDetailsProvided(Case('test'), 'addSuccess', ['foo'])

    def test_addUnexpectedSuccess(self):

        class Case(TestCase):

            def test(this):
                this.addDetail('foo', self.get_content())
                raise testcase._UnexpectedSuccess()
        self.assertDetailsProvided(Case('test'), 'addUnexpectedSuccess', ['foo'])

    def test_addDetails_from_Mismatch(self):
        content = self.get_content()

        class Mismatch:

            def describe(self):
                return 'Mismatch'

            def get_details(self):
                return {'foo': content}

        class Matcher:

            def match(self, thing):
                return Mismatch()

            def __str__(self):
                return 'a description'

        class Case(TestCase):

            def test(self):
                self.assertThat('foo', Matcher())
        self.assertDetailsProvided(Case('test'), 'addFailure', ['foo', 'traceback'])

    def test_multiple_addDetails_from_Mismatch(self):
        content = self.get_content()

        class Mismatch:

            def describe(self):
                return 'Mismatch'

            def get_details(self):
                return {'foo': content, 'bar': content}

        class Matcher:

            def match(self, thing):
                return Mismatch()

            def __str__(self):
                return 'a description'

        class Case(TestCase):

            def test(self):
                self.assertThat('foo', Matcher())
        self.assertDetailsProvided(Case('test'), 'addFailure', ['bar', 'foo', 'traceback'])

    def test_addDetails_with_same_name_as_key_from_get_details(self):
        content = self.get_content()

        class Mismatch:

            def describe(self):
                return 'Mismatch'

            def get_details(self):
                return {'foo': content}

        class Matcher:

            def match(self, thing):
                return Mismatch()

            def __str__(self):
                return 'a description'

        class Case(TestCase):

            def test(self):
                self.addDetail('foo', content)
                self.assertThat('foo', Matcher())
        self.assertDetailsProvided(Case('test'), 'addFailure', ['foo', 'foo-1', 'traceback'])

    def test_addDetailUniqueName_works(self):
        content = self.get_content()

        class Case(TestCase):

            def test(self):
                self.addDetailUniqueName('foo', content)
                self.addDetailUniqueName('foo', content)
        self.assertDetailsProvided(Case('test'), 'addSuccess', ['foo', 'foo-1'])