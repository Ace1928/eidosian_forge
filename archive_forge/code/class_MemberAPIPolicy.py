from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
class MemberAPIPolicy(APIPolicyBase):

    def __init__(self, context, image, target=None, enforcer=None):
        self._context = context
        self._image = image
        if not target:
            self._target = self._build_target()
        self.enforcer = enforcer or policy.Enforcer()
        super(MemberAPIPolicy, self).__init__(context, target=self._target, enforcer=self.enforcer)

    def _build_target(self):
        target = {'project_id': self._context.project_id}
        if self._image:
            target = policy.ImageTarget(self._image)
        return target

    def _enforce(self, rule_name):
        ImageAPIPolicy(self._context, self._image, enforcer=self.enforcer).get_image()
        super(MemberAPIPolicy, self)._enforce(rule_name)

    def get_members(self):
        self._enforce('get_members')

    def get_member(self):
        self._enforce('get_member')

    def delete_member(self):
        self._enforce('delete_member')

    def modify_member(self):
        self._enforce('modify_member')

    def add_member(self):
        self._enforce('add_member')