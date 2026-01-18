import builtins
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.tests.unit import utils as test_utils
class CommonParamsForListTest(test_utils.BaseTestCase):

    def setUp(self):
        super(CommonParamsForListTest, self).setUp()
        self.args = mock.Mock(marker=None, limit=None, sort_key=None, sort_dir=None, detail=False, fields=None, spec=True)
        self.expected_params = {'detail': False}

    def test_nothing_set(self):
        self.assertEqual(self.expected_params, utils.common_params_for_list(self.args, [], []))

    def test_marker_and_limit(self):
        self.args.marker = 'foo'
        self.args.limit = 42
        self.expected_params.update({'marker': 'foo', 'limit': 42})
        self.assertEqual(self.expected_params, utils.common_params_for_list(self.args, [], []))

    def test_invalid_limit(self):
        self.args.limit = -42
        self.assertRaises(exc.CommandError, utils.common_params_for_list, self.args, [], [])

    def test_sort_key_and_sort_dir(self):
        self.args.sort_key = 'field'
        self.args.sort_dir = 'desc'
        self.expected_params.update({'sort_key': 'field', 'sort_dir': 'desc'})
        self.assertEqual(self.expected_params, utils.common_params_for_list(self.args, ['field'], []))

    def test_sort_key_allows_label(self):
        self.args.sort_key = 'Label'
        self.expected_params.update({'sort_key': 'field'})
        self.assertEqual(self.expected_params, utils.common_params_for_list(self.args, ['field', 'field2'], ['Label', 'Label2']))

    def test_sort_key_invalid(self):
        self.args.sort_key = 'something'
        self.assertRaises(exc.CommandError, utils.common_params_for_list, self.args, ['field', 'field2'], [])

    def test_sort_dir_invalid(self):
        self.args.sort_dir = 'something'
        self.assertRaises(exc.CommandError, utils.common_params_for_list, self.args, [], [])

    def test_detail(self):
        self.args.detail = True
        self.expected_params['detail'] = True
        self.assertEqual(self.expected_params, utils.common_params_for_list(self.args, [], []))

    def test_fields(self):
        self.args.fields = [['a', 'b', 'c']]
        self.expected_params.update({'fields': ['a', 'b', 'c']})
        self.assertEqual(self.expected_params, utils.common_params_for_list(self.args, [], []))