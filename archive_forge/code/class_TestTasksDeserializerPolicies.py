import datetime
import http.client as http
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.tasks
from glance.common import timeutils
import glance.domain
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestTasksDeserializerPolicies(test_utils.BaseTestCase):

    def setUp(self):
        super(TestTasksDeserializerPolicies, self).setUp()
        self.policy = unit_test_utils.FakePolicyEnforcer()
        self.deserializer = glance.api.v2.tasks.RequestDeserializer(schema=None, policy_engine=self.policy)
    bad_path = '/tasks?limit=NaN'

    def test_access_index_authorized_bad_query_string(self):
        """Allow access, fail with 400"""
        rules = {'tasks_api_access': True, 'get_tasks': True}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request(self.bad_path)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.index, request)

    def test_access_index_unauthorized(self):
        """Disallow access with bad request, fail with 403"""
        rules = {'tasks_api_access': False, 'get_tasks': True}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request(self.bad_path)
        self.assertRaises(webob.exc.HTTPForbidden, self.deserializer.index, request)
    bad_task = {'typo': 'import', 'input': {'import_from': 'fake'}}

    def test_access_create_authorized_bad_format(self):
        """Allow access, fail with 400"""
        rules = {'tasks_api_access': True, 'add_task': True}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes(self.bad_task)
        self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.create, request)

    def test_access_create_unauthorized(self):
        """Disallow access with bad request, fail with 403"""
        rules = {'tasks_api_access': False, 'add_task': True}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        request.body = jsonutils.dump_as_bytes(self.bad_task)
        self.assertRaises(webob.exc.HTTPForbidden, self.deserializer.create, request)