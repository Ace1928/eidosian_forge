import http.client as http
import ddt
import webob
from oslo_serialization import jsonutils
from glance.api.middleware import version_negotiation
from glance.api import versions
from glance.tests.unit import base
def _assert_version_is_negotiated(self, version_id):
    request = webob.Request.blank('/%s/images' % version_id)
    self.middleware.process_request(request)
    major = version_id.split('.', 1)[0]
    expected = '/%s/images' % major
    self.assertEqual(expected, request.path_info)