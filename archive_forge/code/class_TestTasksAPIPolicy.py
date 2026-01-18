from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
class TestTasksAPIPolicy(APIPolicyBase):

    def setUp(self):
        super(TestTasksAPIPolicy, self).setUp()
        self.enforcer = mock.MagicMock()
        self.context = mock.MagicMock()
        self.policy = policy.TasksAPIPolicy(self.context, enforcer=self.enforcer)

    def test_tasks_api_access(self):
        self.policy.tasks_api_access()
        self.enforcer.enforce.assert_called_once_with(self.context, 'tasks_api_access', mock.ANY)