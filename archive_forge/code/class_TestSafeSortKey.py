import collections
import re
import testtools
from neutron_lib.tests import _base as base
from neutron_lib.utils import helpers
class TestSafeSortKey(base.BaseTestCase):

    def test_safe_sort_key(self):
        data1 = {'k1': 'v1', 'k2': 'v2'}
        data2 = {'k2': 'v2', 'k1': 'v1'}
        self.assertEqual(helpers.safe_sort_key(data1), helpers.safe_sort_key(data2))

    def _create_dict_from_list(self, list_data):
        d = collections.defaultdict(list)
        for k, v in list_data:
            d[k].append(v)
        return d

    def test_safe_sort_key_mapping_ne(self):
        list1 = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
        data1 = self._create_dict_from_list(list1)
        list2 = [('yellow', 3), ('blue', 4), ('yellow', 1), ('blue', 2), ('red', 1)]
        data2 = self._create_dict_from_list(list2)
        self.assertNotEqual(helpers.safe_sort_key(data1), helpers.safe_sort_key(data2))

    def test_safe_sort_key_mapping(self):
        list1 = [('yellow', 1), ('blue', 2), ('red', 1)]
        data1 = self._create_dict_from_list(list1)
        list2 = [('blue', 2), ('red', 1), ('yellow', 1)]
        data2 = self._create_dict_from_list(list2)
        self.assertEqual(helpers.safe_sort_key(data1), helpers.safe_sort_key(data2))