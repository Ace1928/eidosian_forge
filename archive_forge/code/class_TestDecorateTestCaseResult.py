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
class TestDecorateTestCaseResult(TestCase):

    def setUp(self):
        super().setUp()
        self.log = []

    def make_result(self, result):
        self.log.append(('result', result))
        return LoggingResult(self.log)

    def test___call__(self):
        case = DecorateTestCaseResult(PlaceHolder('foo'), self.make_result)
        case(None)
        case('something')
        self.assertEqual([('result', None), ('tags', set(), set()), ('startTest', case.decorated), ('addSuccess', case.decorated), ('stopTest', case.decorated), ('tags', set(), set()), ('result', 'something'), ('tags', set(), set()), ('startTest', case.decorated), ('addSuccess', case.decorated), ('stopTest', case.decorated), ('tags', set(), set())], self.log)

    def test_run(self):
        case = DecorateTestCaseResult(PlaceHolder('foo'), self.make_result)
        case.run(None)
        case.run('something')
        self.assertEqual([('result', None), ('tags', set(), set()), ('startTest', case.decorated), ('addSuccess', case.decorated), ('stopTest', case.decorated), ('tags', set(), set()), ('result', 'something'), ('tags', set(), set()), ('startTest', case.decorated), ('addSuccess', case.decorated), ('stopTest', case.decorated), ('tags', set(), set())], self.log)

    def test_before_after_hooks(self):
        case = DecorateTestCaseResult(PlaceHolder('foo'), self.make_result, before_run=lambda result: self.log.append('before'), after_run=lambda result: self.log.append('after'))
        case.run(None)
        case(None)
        self.assertEqual([('result', None), 'before', ('tags', set(), set()), ('startTest', case.decorated), ('addSuccess', case.decorated), ('stopTest', case.decorated), ('tags', set(), set()), 'after', ('result', None), 'before', ('tags', set(), set()), ('startTest', case.decorated), ('addSuccess', case.decorated), ('stopTest', case.decorated), ('tags', set(), set()), 'after'], self.log)

    def test_other_attribute(self):
        orig = PlaceHolder('foo')
        orig.thing = 'fred'
        case = DecorateTestCaseResult(orig, self.make_result)
        self.assertEqual('fred', case.thing)
        self.assertRaises(AttributeError, getattr, case, 'other')
        case.other = 'barbara'
        self.assertEqual('barbara', orig.other)
        del case.thing
        self.assertRaises(AttributeError, getattr, orig, 'thing')