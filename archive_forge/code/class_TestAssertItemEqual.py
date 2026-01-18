import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
class TestAssertItemEqual(test_utils.TestCommand):

    def test_assert_normal_item(self):
        expected = ['a', 'b', 'c']
        actual = ['a', 'b', 'c']
        self.assertItemEqual(expected, actual)

    def test_assert_item_with_formattable_columns(self):
        expected = [format_columns.DictColumn({'a': 1, 'b': 2}), format_columns.ListColumn(['x', 'y', 'z'])]
        actual = [format_columns.DictColumn({'a': 1, 'b': 2}), format_columns.ListColumn(['x', 'y', 'z'])]
        self.assertItemEqual(expected, actual)

    def test_assert_item_different_length(self):
        expected = ['a', 'b', 'c']
        actual = ['a', 'b']
        self.assertRaises(AssertionError, self.assertItemEqual, expected, actual)

    def test_assert_item_formattable_columns_vs_legacy_formatter(self):
        expected = [format_columns.DictColumn({'a': 1, 'b': 2}), format_columns.ListColumn(['x', 'y', 'z'])]
        actual = [utils.format_dict({'a': 1, 'b': 2}), utils.format_list(['x', 'y', 'z'])]
        self.assertRaises(AssertionError, self.assertItemEqual, expected, actual)

    def test_assert_item_different_formattable_columns(self):

        class ExceptionColumn(cliff_columns.FormattableColumn):

            def human_readable(self):
                raise Exception('always fail')
        expected = [format_columns.DictColumn({'a': 1, 'b': 2})]
        actual = [ExceptionColumn({'a': 1, 'b': 2})]
        self.assertRaises(AssertionError, self.assertItemEqual, expected, actual)

    def test_assert_list_item(self):
        expected = [['a', 'b', 'c'], [format_columns.DictColumn({'a': 1, 'b': 2}), format_columns.ListColumn(['x', 'y', 'z'])]]
        actual = [['a', 'b', 'c'], [format_columns.DictColumn({'a': 1, 'b': 2}), format_columns.ListColumn(['x', 'y', 'z'])]]
        self.assertListItemEqual(expected, actual)