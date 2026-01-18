import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
class TestFieldWidths(base.TestBase):

    def test(self):
        tf = table.TableFormatter
        self.assertEqual({'a': 1, 'b': 2, 'c': 3, 'd': 10}, tf._field_widths(('a', 'b', 'c', 'd'), '+---+----+-----+------------+'))

    def test_zero(self):
        tf = table.TableFormatter
        self.assertEqual({'a': 0, 'b': 0, 'c': 0}, tf._field_widths(('a', 'b', 'c'), '+--+-++'))

    def test_info(self):
        tf = table.TableFormatter
        self.assertEqual((49, 4), tf._width_info(80, 10))
        self.assertEqual((76, 76), tf._width_info(80, 1))
        self.assertEqual((79, 0), tf._width_info(80, 0))
        self.assertEqual((0, 0), tf._width_info(0, 80))