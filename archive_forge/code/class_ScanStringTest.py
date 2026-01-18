from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ScanStringTest(ParseTestCase):

    def runTest(self):
        from pyparsing import Word, Combine, Suppress, CharsNotIn, nums, StringEnd
        testdata = '\n            <table border="0" cellpadding="3" cellspacing="3" frame="" width="90%">\n                <tr align="left" valign="top">\n                        <td><b>Name</b></td>\n                        <td><b>IP Address</b></td>\n                        <td><b>Location</b></td>\n                </tr>\n                <tr align="left" valign="top" bgcolor="#c7efce">\n                        <td>time-a.nist.gov</td>\n                        <td>129.6.15.28</td>\n                        <td>NIST, Gaithersburg, Maryland</td>\n                </tr>\n                <tr align="left" valign="top">\n                        <td>time-b.nist.gov</td>\n                        <td>129.6.15.29</td>\n                        <td>NIST, Gaithersburg, Maryland</td>\n                </tr>\n                <tr align="left" valign="top" bgcolor="#c7efce">\n                        <td>time-a.timefreq.bldrdoc.gov</td>\n                        <td>132.163.4.101</td>\n                        <td>NIST, Boulder, Colorado</td>\n                </tr>\n                <tr align="left" valign="top">\n                        <td>time-b.timefreq.bldrdoc.gov</td>\n                        <td>132.163.4.102</td>\n                        <td>NIST, Boulder, Colorado</td>\n                </tr>\n                <tr align="left" valign="top" bgcolor="#c7efce">\n                        <td>time-c.timefreq.bldrdoc.gov</td>\n                        <td>132.163.4.103</td>\n                        <td>NIST, Boulder, Colorado</td>\n                </tr>\n            </table>\n            '
        integer = Word(nums)
        ipAddress = Combine(integer + '.' + integer + '.' + integer + '.' + integer)
        tdStart = Suppress('<td>')
        tdEnd = Suppress('</td>')
        timeServerPattern = tdStart + ipAddress('ipAddr') + tdEnd + tdStart + CharsNotIn('<')('loc') + tdEnd
        servers = [srvr.ipAddr for srvr, startloc, endloc in timeServerPattern.scanString(testdata)]
        print_(servers)
        self.assertEqual(servers, ['129.6.15.28', '129.6.15.29', '132.163.4.101', '132.163.4.102', '132.163.4.103'], 'failed scanString()')
        foundStringEnds = [r for r in StringEnd().scanString('xyzzy')]
        print_(foundStringEnds)
        self.assertTrue(foundStringEnds, 'Failed to find StringEnd in scanString')