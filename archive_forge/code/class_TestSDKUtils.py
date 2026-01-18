import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
class TestSDKUtils(test_utils.TestCase):

    def setUp(self):
        super(TestSDKUtils, self).setUp()

    def _test_get_osc_show_columns_for_sdk_resource(self, sdk_resource, column_map, expected_display_columns, expected_attr_columns):
        display_columns, attr_columns = utils.get_osc_show_columns_for_sdk_resource(sdk_resource, column_map)
        self.assertEqual(expected_display_columns, display_columns)
        self.assertEqual(expected_attr_columns, attr_columns)

    def test_get_osc_show_columns_for_sdk_resource_empty(self):
        self._test_get_osc_show_columns_for_sdk_resource({}, {}, tuple(), tuple())

    def test_get_osc_show_columns_for_sdk_resource_empty_map(self):
        self._test_get_osc_show_columns_for_sdk_resource({'foo': 'foo1'}, {}, ('foo',), ('foo',))

    def test_get_osc_show_columns_for_sdk_resource_empty_data(self):
        self._test_get_osc_show_columns_for_sdk_resource({}, {'foo': 'foo_map'}, ('foo_map',), ('foo_map',))

    def test_get_osc_show_columns_for_sdk_resource_map(self):
        self._test_get_osc_show_columns_for_sdk_resource({'foo': 'foo1'}, {'foo': 'foo_map'}, ('foo_map',), ('foo',))

    def test_get_osc_show_columns_for_sdk_resource_map_dup(self):
        self._test_get_osc_show_columns_for_sdk_resource({'foo': 'foo1', 'foo_map': 'foo1'}, {'foo': 'foo_map'}, ('foo_map',), ('foo',))

    def test_get_osc_show_columns_for_sdk_resource_map_full(self):
        self._test_get_osc_show_columns_for_sdk_resource({'foo': 'foo1', 'bar': 'bar1'}, {'foo': 'foo_map', 'new': 'bar'}, ('bar', 'foo_map'), ('bar', 'foo'))