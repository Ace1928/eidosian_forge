import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
class EndpointFilterTestCase(test_v3.RestfulTestCase):

    def setUp(self):
        super(EndpointFilterTestCase, self).setUp()
        self.default_request_url = '/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.default_domain_project_id, 'endpoint_id': self.endpoint_id}