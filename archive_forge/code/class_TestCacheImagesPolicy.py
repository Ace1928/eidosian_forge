from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.image_cache import prefetcher
from glance.tests import functional
class TestCacheImagesPolicy(functional.SynchronousAPIBase):

    def setUp(self):
        super(TestCacheImagesPolicy, self).setUp()
        self.policy = policy.Enforcer(suppress_deprecation_warnings=True)

    def set_policy_rules(self, rules):
        self.policy.set_rules(oslo_policy.policy.Rules.from_dict(rules), overwrite=True)

    def start_server(self):
        with mock.patch.object(policy, 'Enforcer') as mock_enf:
            mock_enf.return_value = self.policy
            super(TestCacheImagesPolicy, self).start_server(enable_cache=True)

    def _create_upload_and_cache(self, cache_image=False, expected_code=200):
        image_id = self._create_and_upload()
        path = '/v2/queued_images/%s' % image_id
        response = self.api_put(path)
        self.assertEqual(expected_code, response.status_code)
        if cache_image:
            cache_prefetcher = prefetcher.Prefetcher()
            cache_prefetcher.run()
        return image_id

    def test_queued_images(self):
        self.start_server()
        self._create_upload_and_cache(expected_code=200)
        self.set_policy_rules({'manage_image_cache': '!', 'add_image': '', 'upload_image': ''})
        self._create_upload_and_cache(expected_code=403)

    def test_get_queued_images(self):
        self.start_server()
        image_id = self._create_upload_and_cache()
        path = '/v2/queued_images'
        response = self.api_get(path)
        self.assertEqual(200, response.status_code)
        output = response.json
        self.assertIn(image_id, output['queued_images'])
        self.set_policy_rules({'manage_image_cache': '!'})
        response = self.api_get(path)
        self.assertEqual(403, response.status_code)

    def test_delete_queued_image(self):
        self.start_server()
        image_id = self._create_upload_and_cache()
        second_image_id = self._create_upload_and_cache()
        path = '/v2/queued_images/%s' % image_id
        response = self.api_delete(path)
        self.assertEqual(200, response.status_code)
        path = '/v2/queued_images'
        response = self.api_get(path)
        output = response.json
        self.assertNotIn(image_id, output['queued_images'])
        self.set_policy_rules({'manage_image_cache': '!'})
        path = '/v2/queued_images/%s' % second_image_id
        response = self.api_delete(path)
        self.assertEqual(403, response.status_code)

    def test_delete_queued_images(self):
        self.start_server()
        self._create_upload_and_cache()
        self._create_upload_and_cache()
        path = '/v2/queued_images'
        response = self.api_delete(path)
        self.assertEqual(200, response.status_code)
        path = '/v2/queued_images'
        response = self.api_get(path)
        output = response.json
        self.assertEqual([], output['queued_images'])
        image_id = self._create_upload_and_cache()
        self.set_policy_rules({'manage_image_cache': '!'})
        path = '/v2/queued_images'
        response = self.api_delete(path)
        self.assertEqual(403, response.status_code)
        self.set_policy_rules({'manage_image_cache': ''})
        path = '/v2/queued_images'
        response = self.api_get(path)
        output = response.json
        self.assertIn(image_id, output['queued_images'])

    def test_get_cached_images(self):
        self.start_server()
        image_id = self._create_upload_and_cache(cache_image=True)
        path = '/v2/cached_images'
        response = self.api_get(path)
        self.assertEqual(200, response.status_code)
        output = response.json
        self.assertEqual(image_id, output['cached_images'][0]['image_id'])
        self.set_policy_rules({'manage_image_cache': '!'})
        response = self.api_get(path)
        self.assertEqual(403, response.status_code)

    def test_delete_cached_image(self):
        self.start_server()
        image_id = self._create_upload_and_cache(cache_image=True)
        second_image_id = self._create_upload_and_cache(cache_image=True)
        path = '/v2/cached_images/%s' % image_id
        response = self.api_delete(path)
        self.assertEqual(200, response.status_code)
        path = '/v2/cached_images'
        response = self.api_get(path)
        output = response.json
        self.assertEqual(1, len(output['cached_images']))
        self.set_policy_rules({'manage_image_cache': '!'})
        path = '/v2/cached_images/%s' % second_image_id
        response = self.api_delete(path)
        self.assertEqual(403, response.status_code)

    def test_delete_cached_images(self):
        self.start_server()
        self._create_upload_and_cache(cache_image=True)
        self._create_upload_and_cache(cache_image=True)
        path = '/v2/cached_images'
        response = self.api_delete(path)
        self.assertEqual(200, response.status_code)
        path = '/v2/cached_images'
        response = self.api_get(path)
        output = response.json
        self.assertEqual(0, len(output['cached_images']))
        self._create_upload_and_cache(cache_image=True)
        self.set_policy_rules({'manage_image_cache': '!'})
        path = '/v2/cached_images'
        response = self.api_delete(path)
        self.assertEqual(403, response.status_code)
        self.set_policy_rules({'manage_image_cache': ''})
        path = '/v2/cached_images'
        response = self.api_get(path)
        output = response.json
        self.assertEqual(1, len(output['cached_images']))