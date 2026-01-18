import collections
import io
import sys
from unittest import mock
import ddt
from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient.tests.unit import utils as test_utils
from cinderclient import utils
class PrintListTestCase(test_utils.TestCase):

    def test_print_list_with_list(self):
        Row = collections.namedtuple('Row', ['a', 'b'])
        to_print = [Row(a=3, b=4), Row(a=1, b=2)]
        with CaptureStdout() as cso:
            shell_utils.print_list(to_print, ['a', 'b'])
        self.assertEqual('+---+---+\n| a | b |\n+---+---+\n| 1 | 2 |\n| 3 | 4 |\n+---+---+\n', cso.read())

    def test_print_list_with_None_data(self):
        Row = collections.namedtuple('Row', ['a', 'b'])
        to_print = [Row(a=3, b=None), Row(a=1, b=2)]
        with CaptureStdout() as cso:
            shell_utils.print_list(to_print, ['a', 'b'])
        self.assertEqual('+---+---+\n| a | b |\n+---+---+\n| 1 | 2 |\n| 3 | - |\n+---+---+\n', cso.read())

    def test_print_list_with_list_sortby(self):
        Row = collections.namedtuple('Row', ['a', 'b'])
        to_print = [Row(a=4, b=3), Row(a=2, b=1)]
        with CaptureStdout() as cso:
            shell_utils.print_list(to_print, ['a', 'b'], sortby_index=1)
        self.assertEqual('+---+---+\n| a | b |\n+---+---+\n| 2 | 1 |\n| 4 | 3 |\n+---+---+\n', cso.read())

    def test_print_list_with_list_no_sort(self):
        Row = collections.namedtuple('Row', ['a', 'b'])
        to_print = [Row(a=3, b=4), Row(a=1, b=2)]
        with CaptureStdout() as cso:
            shell_utils.print_list(to_print, ['a', 'b'], sortby_index=None)
        self.assertEqual('+---+---+\n| a | b |\n+---+---+\n| 3 | 4 |\n| 1 | 2 |\n+---+---+\n', cso.read())

    def test_print_list_with_generator(self):
        Row = collections.namedtuple('Row', ['a', 'b'])

        def gen_rows():
            for row in [Row(a=1, b=2), Row(a=3, b=4)]:
                yield row
        with CaptureStdout() as cso:
            shell_utils.print_list(gen_rows(), ['a', 'b'])
        self.assertEqual('+---+---+\n| a | b |\n+---+---+\n| 1 | 2 |\n| 3 | 4 |\n+---+---+\n', cso.read())

    def test_print_list_with_return(self):
        Row = collections.namedtuple('Row', ['a', 'b'])
        to_print = [Row(a=3, b='a\r'), Row(a=1, b='c\rd')]
        with CaptureStdout() as cso:
            shell_utils.print_list(to_print, ['a', 'b'])
        self.assertEqual('+---+-----+\n| a | b   |\n+---+-----+\n| 1 | c d |\n| 3 | a   |\n+---+-----+\n', cso.read())