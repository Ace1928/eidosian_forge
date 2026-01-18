import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
class EndpointGroupsTestMixin(object):

    def check_endpoint_group(self, endpoint_group, endpoint_group_ref=None):
        self.assertIsNotNone(endpoint_group.id)
        self.assertIn('self', endpoint_group.links)
        self.assertIn('/endpoint_groups/' + endpoint_group.id, endpoint_group.links['self'])
        if endpoint_group_ref:
            self.assertEqual(endpoint_group_ref['name'], endpoint_group.name)
            self.assertEqual(endpoint_group_ref['filters'], endpoint_group.filters)
            if hasattr(endpoint_group_ref, 'description'):
                self.assertEqual(endpoint_group_ref['description'], endpoint_group.description)
        else:
            self.assertIsNotNone(endpoint_group.name)
            self.assertIsNotNone(endpoint_group.filters)