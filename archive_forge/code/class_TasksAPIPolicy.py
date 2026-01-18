from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
class TasksAPIPolicy(APIPolicyBase):

    def __init__(self, context, target=None, enforcer=None):
        self._context = context
        self._target = target or {}
        self.enforcer = enforcer or policy.Enforcer()
        super(TasksAPIPolicy, self).__init__(context, target=self._target, enforcer=self.enforcer)

    def tasks_api_access(self):
        self._enforce('tasks_api_access')