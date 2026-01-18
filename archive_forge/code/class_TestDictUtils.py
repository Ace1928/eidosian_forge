import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
class TestDictUtils(base.BaseTestCase):

    def test_dict2str(self):
        dic = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        expected = 'key1=value1,key2=value2,key3=value3'
        self.assertEqual(expected, helpers.dict2str(dic))

    def test_str2dict(self):
        string = 'key1=value1,key2=value2,key3=value3'
        expected = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        self.assertEqual(expected, helpers.str2dict(string))

    def test_dict_str_conversion(self):
        dic = {'key1': 'value1', 'key2': 'value2'}
        self.assertEqual(dic, helpers.str2dict(helpers.dict2str(dic)))

    def test_diff_list_of_dict(self):
        old_list = [{'key1': 'value1'}, {'key2': 'value2'}, {'key3': 'value3'}]
        new_list = [{'key1': 'value1'}, {'key2': 'value2'}, {'key4': 'value4'}]
        added, removed = helpers.diff_list_of_dict(old_list, new_list)
        self.assertEqual(added, [dict(key4='value4')])
        self.assertEqual(removed, [dict(key3='value3')])