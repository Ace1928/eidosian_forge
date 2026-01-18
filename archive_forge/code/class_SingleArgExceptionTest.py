from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class SingleArgExceptionTest(ParseTestCase):

    def runTest(self):
        from pyparsing import ParseBaseException, ParseFatalException
        msg = ''
        raisedMsg = ''
        testMessage = 'just one arg'
        try:
            raise ParseFatalException(testMessage)
        except ParseBaseException as pbe:
            print_('Received expected exception:', pbe)
            raisedMsg = pbe.msg
            self.assertEqual(raisedMsg, testMessage, 'Failed to get correct exception message')