from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseResultsNamesInGroupWithDictTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        from pyparsing import pyparsing_common as ppc
        key = ppc.identifier()
        value = ppc.integer()
        lat = ppc.real()
        long = ppc.real()
        EQ = pp.Suppress('=')
        data = lat('lat') + long('long') + pp.Dict(pp.OneOrMore(pp.Group(key + EQ + value)))
        site = pp.QuotedString('"')('name') + pp.Group(data)('data')
        test_string = '"Golden Gate Bridge" 37.819722 -122.478611 height=746 span=4200'
        site.runTests(test_string)
        a, aEnd = pp.makeHTMLTags('a')
        attrs = a.parseString("<a href='blah'>")
        print_(attrs.dump())
        self.assertEqual(attrs.startA.href, 'blah')
        self.assertEqual(attrs.asDict(), {'startA': {'href': 'blah', 'tag': 'a', 'empty': False}, 'href': 'blah', 'tag': 'a', 'empty': False})