import http.client as http
import ddt
import webob
from oslo_serialization import jsonutils
from glance.api.middleware import version_negotiation
from glance.api import versions
from glance.tests.unit import base
@ddt.ddt
class VersionsAndNegotiationTest(VersionNegotiationTest, VersionsTest):
    """
    Test that versions mentioned in the versions response are correctly
    negotiated.
    """

    def _get_list_of_version_ids(self, status):
        request = webob.Request.blank('/')
        request.accept = 'application/json'
        response = versions.Controller().index(request)
        v_list = jsonutils.loads(response.body)['versions']
        return [v['id'] for v in v_list if v['status'] == status]

    def _assert_version_is_negotiated(self, version_id):
        request = webob.Request.blank('/%s/images' % version_id)
        self.middleware.process_request(request)
        major = version_id.split('.', 1)[0]
        expected = '/%s/images' % major
        self.assertEqual(expected, request.path_info)
    cache = '/var/cache'
    multistore = 'slow:one,fast:two'
    combos = ((None, None), (None, multistore), (cache, None), (cache, multistore))

    @ddt.data(*combos)
    @ddt.unpack
    def test_current_is_negotiated(self, cache, multistore):
        self.config(enabled_backends=multistore)
        self.config(image_cache_dir=cache)
        to_check = self._get_list_of_version_ids('CURRENT')
        self.assertTrue(to_check)
        for version_id in to_check:
            self._assert_version_is_negotiated(version_id)

    @ddt.data(*combos)
    @ddt.unpack
    def test_supported_is_negotiated(self, cache, multistore):
        self.config(enabled_backends=multistore)
        self.config(image_cache_dir=cache)
        to_check = self._get_list_of_version_ids('SUPPORTED')
        for version_id in to_check:
            self._assert_version_is_negotiated(version_id)

    @ddt.data(*combos)
    @ddt.unpack
    def test_deprecated_is_negotiated(self, cache, multistore):
        self.config(enabled_backends=multistore)
        self.config(image_cache_dir=cache)
        to_check = self._get_list_of_version_ids('DEPRECATED')
        for version_id in to_check:
            self._assert_version_is_negotiated(version_id)

    @ddt.data(*combos)
    @ddt.unpack
    def test_experimental_is_negotiated(self, cache, multistore):
        self.config(enabled_backends=multistore)
        self.config(image_cache_dir=cache)
        to_check = self._get_list_of_version_ids('EXPERIMENTAL')
        for version_id in to_check:
            self._assert_version_is_negotiated(version_id)