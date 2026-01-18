from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class SumParseResultsTest(ParseTestCase):

    def runTest(self):
        samplestr1 = 'garbage;DOB 10-10-2010;more garbage\nID PARI12345678;more garbage'
        samplestr2 = 'garbage;ID PARI12345678;more garbage\nDOB 10-10-2010;more garbage'
        samplestr3 = 'garbage;DOB 10-10-2010'
        samplestr4 = 'garbage;ID PARI12345678;more garbage- I am cool'
        res1 = 'ID:PARI12345678 DOB:10-10-2010 INFO:'
        res2 = 'ID:PARI12345678 DOB:10-10-2010 INFO:'
        res3 = 'ID: DOB:10-10-2010 INFO:'
        res4 = 'ID:PARI12345678 DOB: INFO: I am cool'
        from pyparsing import Regex, Word, alphanums, restOfLine
        dob_ref = 'DOB' + Regex('\\d{2}-\\d{2}-\\d{4}')('dob')
        id_ref = 'ID' + Word(alphanums, exact=12)('id')
        info_ref = '-' + restOfLine('info')
        person_data = dob_ref | id_ref | info_ref
        tests = (samplestr1, samplestr2, samplestr3, samplestr4)
        results = (res1, res2, res3, res4)
        for test, expected in zip(tests, results):
            person = sum(person_data.searchString(test))
            result = 'ID:%s DOB:%s INFO:%s' % (person.id, person.dob, person.info)
            print_(test)
            print_(expected)
            print_(result)
            for pd in person_data.searchString(test):
                print_(pd.dump())
            print_()
            self.assertEqual(expected, result, "Failed to parse '%s' correctly, \nexpected '%s', got '%s'" % (test, expected, result))