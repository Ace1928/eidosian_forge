from openstack import exceptions
from openstack.tests.functional import base
class TestRangeSearch(base.BaseFunctionalTest):

    def _filter_m1_flavors(self, results):
        """The m1 flavors are the original devstack flavors"""
        new_results = []
        for flavor in results:
            if flavor['name'].startswith('m1.'):
                new_results.append(flavor)
        return new_results

    def test_range_search_bad_range(self):
        flavors = self.user_cloud.list_flavors(get_extra=False)
        self.assertRaises(exceptions.SDKException, self.user_cloud.range_search, flavors, {'ram': '<1a0'})

    def test_range_search_exact(self):
        flavors = self.user_cloud.list_flavors(get_extra=False)
        result = self.user_cloud.range_search(flavors, {'ram': '4096'})
        self.assertIsInstance(result, list)
        result = self._filter_m1_flavors(result)
        self.assertEqual(1, len(result))
        self.assertEqual('m1.medium', result[0]['name'])

    def test_range_search_min(self):
        flavors = self.user_cloud.list_flavors(get_extra=False)
        result = self.user_cloud.range_search(flavors, {'ram': 'MIN'})
        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))
        self.assertIn(result[0]['name'], ('cirros256', 'm1.tiny'))

    def test_range_search_max(self):
        flavors = self.user_cloud.list_flavors(get_extra=False)
        result = self.user_cloud.range_search(flavors, {'ram': 'MAX'})
        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))
        self.assertEqual('m1.xlarge', result[0]['name'])

    def test_range_search_lt(self):
        flavors = self.user_cloud.list_flavors(get_extra=False)
        result = self.user_cloud.range_search(flavors, {'ram': '<1024'})
        self.assertIsInstance(result, list)
        result = self._filter_m1_flavors(result)
        self.assertEqual(1, len(result))
        self.assertEqual('m1.tiny', result[0]['name'])

    def test_range_search_gt(self):
        flavors = self.user_cloud.list_flavors(get_extra=False)
        result = self.user_cloud.range_search(flavors, {'ram': '>4096'})
        self.assertIsInstance(result, list)
        result = self._filter_m1_flavors(result)
        self.assertEqual(2, len(result))
        flavor_names = [r['name'] for r in result]
        self.assertIn('m1.large', flavor_names)
        self.assertIn('m1.xlarge', flavor_names)

    def test_range_search_le(self):
        flavors = self.user_cloud.list_flavors(get_extra=False)
        result = self.user_cloud.range_search(flavors, {'ram': '<=4096'})
        self.assertIsInstance(result, list)
        result = self._filter_m1_flavors(result)
        self.assertEqual(3, len(result))
        flavor_names = [r['name'] for r in result]
        self.assertIn('m1.tiny', flavor_names)
        self.assertIn('m1.small', flavor_names)
        self.assertIn('m1.medium', flavor_names)

    def test_range_search_ge(self):
        flavors = self.user_cloud.list_flavors(get_extra=False)
        result = self.user_cloud.range_search(flavors, {'ram': '>=4096'})
        self.assertIsInstance(result, list)
        result = self._filter_m1_flavors(result)
        self.assertEqual(3, len(result))
        flavor_names = [r['name'] for r in result]
        self.assertIn('m1.medium', flavor_names)
        self.assertIn('m1.large', flavor_names)
        self.assertIn('m1.xlarge', flavor_names)

    def test_range_search_multi_1(self):
        flavors = self.user_cloud.list_flavors(get_extra=False)
        result = self.user_cloud.range_search(flavors, {'ram': 'MIN', 'vcpus': 'MIN'})
        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))
        self.assertIn(result[0]['name'], ('cirros256', 'm1.tiny'))

    def test_range_search_multi_2(self):
        flavors = self.user_cloud.list_flavors(get_extra=False)
        result = self.user_cloud.range_search(flavors, {'ram': '<1024', 'vcpus': 'MIN'})
        self.assertIsInstance(result, list)
        result = self._filter_m1_flavors(result)
        self.assertEqual(1, len(result))
        flavor_names = [r['name'] for r in result]
        self.assertIn('m1.tiny', flavor_names)

    def test_range_search_multi_3(self):
        flavors = self.user_cloud.list_flavors(get_extra=False)
        result = self.user_cloud.range_search(flavors, {'ram': '>=4096', 'vcpus': '<6'})
        self.assertIsInstance(result, list)
        result = self._filter_m1_flavors(result)
        self.assertEqual(2, len(result))
        flavor_names = [r['name'] for r in result]
        self.assertIn('m1.medium', flavor_names)
        self.assertIn('m1.large', flavor_names)

    def test_range_search_multi_4(self):
        flavors = self.user_cloud.list_flavors(get_extra=False)
        result = self.user_cloud.range_search(flavors, {'ram': '>=4096', 'vcpus': 'MAX'})
        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))
        self.assertEqual('m1.xlarge', result[0]['name'])