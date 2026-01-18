from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class HTMLStripperTest(ParseTestCase):

    def runTest(self):
        from pyparsing import pyparsing_common, originalTextFor, OneOrMore, Word, printables
        sample = '\n        <html>\n        Here is some sample <i>HTML</i> text.\n        </html>\n        '
        read_everything = originalTextFor(OneOrMore(Word(printables)))
        read_everything.addParseAction(pyparsing_common.stripHTMLTags)
        result = read_everything.parseString(sample)
        self.assertEqual(result[0].strip(), 'Here is some sample HTML text.')