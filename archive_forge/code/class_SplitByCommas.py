import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
class SplitByCommas(test_base.BaseTestCase):

    def test_not_closed_quotes(self):
        self.assertRaises(ValueError, strutils.split_by_commas, '"ab","b""')

    def test_no_comma_before_opening_quotes(self):
        self.assertRaises(ValueError, strutils.split_by_commas, '"ab""b"')

    def test_quote_inside_unquoted(self):
        self.assertRaises(ValueError, strutils.split_by_commas, 'a"b,cd')

    def check(self, expect, input):
        self.assertEqual(expect, strutils.split_by_commas(input))

    def test_plain(self):
        self.check(['a,b', 'ac'], '"a,b",ac')

    def test_with_backslash_inside_quoted(self):
        self.check(['abc"', 'de', 'fg,h', 'klm\\', '"nop'], '"abc\\"","de","fg,h","klm\\\\","\\"nop"')

    def test_with_backslash_inside_unquoted(self):
        self.check(['a\\bc', 'de'], 'a\\bc,de')

    def test_with_escaped_quotes_in_row_inside_quoted(self):
        self.check(['a"b""c', 'd'], '"a\\"b\\"\\"c",d')