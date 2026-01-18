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
class TestTasksControllerPolicies(base.IsolatedUnitTest):

    def setUp(self):
        super(TestTasksControllerPolicies, self).setUp()
        self.db = unit_test_utils.FakeDB()
        self.policy = unit_test_utils.FakePolicyEnforcer()
        self.controller = glance.api.v2.tasks.TasksController(self.db, self.policy)

    def test_access_get_unauthorized(self):
        rules = {'tasks_api_access': False, 'get_task': True}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.get, request, task_id=UUID2)

    def test_delete(self):
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPMethodNotAllowed, self.controller.delete, request, 'fake_id')

    def test_access_delete_unauthorized(self):
        rules = {'tasks_api_access': False}
        self.policy.set_rules(rules)
        request = unit_test_utils.get_fake_request()
        self.assertRaises(webob.exc.HTTPForbidden, self.controller.delete, request, 'fake_id')