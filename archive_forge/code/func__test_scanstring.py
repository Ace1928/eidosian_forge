import sys
from unittest import TestCase
import simplejson as json
import simplejson.decoder
from simplejson.compat import b, PY3
def _test_scanstring(self, scanstring):
    if sys.maxunicode == 65535:
        self.assertEqual(scanstring(u'"z𝄠x"', 1, None, True), (u'z𝄠x', 6))
    else:
        self.assertEqual(scanstring(u'"z𝄠x"', 1, None, True), (u'z𝄠x', 5))
    self.assertEqual(scanstring('"\\u007b"', 1, None, True), (u'{', 8))
    self.assertEqual(scanstring('"A JSON payload should be an object or array, not a string."', 1, None, True), (u'A JSON payload should be an object or array, not a string.', 60))
    self.assertEqual(scanstring('["Unclosed array"', 2, None, True), (u'Unclosed array', 17))
    self.assertEqual(scanstring('["extra comma",]', 2, None, True), (u'extra comma', 14))
    self.assertEqual(scanstring('["double extra comma",,]', 2, None, True), (u'double extra comma', 21))
    self.assertEqual(scanstring('["Comma after the close"],', 2, None, True), (u'Comma after the close', 24))
    self.assertEqual(scanstring('["Extra close"]]', 2, None, True), (u'Extra close', 14))
    self.assertEqual(scanstring('{"Extra comma": true,}', 2, None, True), (u'Extra comma', 14))
    self.assertEqual(scanstring('{"Extra value after close": true} "misplaced quoted value"', 2, None, True), (u'Extra value after close', 26))
    self.assertEqual(scanstring('{"Illegal expression": 1 + 2}', 2, None, True), (u'Illegal expression', 21))
    self.assertEqual(scanstring('{"Illegal invocation": alert()}', 2, None, True), (u'Illegal invocation', 21))
    self.assertEqual(scanstring('{"Numbers cannot have leading zeroes": 013}', 2, None, True), (u'Numbers cannot have leading zeroes', 37))
    self.assertEqual(scanstring('{"Numbers cannot be hex": 0x14}', 2, None, True), (u'Numbers cannot be hex', 24))
    self.assertEqual(scanstring('[[[[[[[[[[[[[[[[[[[["Too deep"]]]]]]]]]]]]]]]]]]]]', 21, None, True), (u'Too deep', 30))
    self.assertEqual(scanstring('{"Missing colon" null}', 2, None, True), (u'Missing colon', 16))
    self.assertEqual(scanstring('{"Double colon":: null}', 2, None, True), (u'Double colon', 15))
    self.assertEqual(scanstring('{"Comma instead of colon", null}', 2, None, True), (u'Comma instead of colon', 25))
    self.assertEqual(scanstring('["Colon instead of comma": false]', 2, None, True), (u'Colon instead of comma', 25))
    self.assertEqual(scanstring('["Bad value", truth]', 2, None, True), (u'Bad value', 12))
    for c in map(chr, range(0, 31)):
        self.assertEqual(scanstring(c + '"', 0, None, False), (c, 2))
        self.assertRaises(ValueError, scanstring, c + '"', 0, None, True)
    self.assertRaises(ValueError, scanstring, '', 0, None, True)
    self.assertRaises(ValueError, scanstring, 'a', 0, None, True)
    self.assertRaises(ValueError, scanstring, '\\', 0, None, True)
    self.assertRaises(ValueError, scanstring, '\\u', 0, None, True)
    self.assertRaises(ValueError, scanstring, '\\u0', 0, None, True)
    self.assertRaises(ValueError, scanstring, '\\u01', 0, None, True)
    self.assertRaises(ValueError, scanstring, '\\u012', 0, None, True)
    self.assertRaises(ValueError, scanstring, '\\u0123', 0, None, True)
    if sys.maxunicode > 65535:
        self.assertRaises(ValueError, scanstring, '\\ud834\\u"', 0, None, True)
        self.assertRaises(ValueError, scanstring, '\\ud834\\x0123"', 0, None, True)
    self.assertRaises(json.JSONDecodeError, scanstring, '\\u-123"', 0, None, True)
    self.assertRaises(json.JSONDecodeError, scanstring, '\\u EDD"', 0, None, True)