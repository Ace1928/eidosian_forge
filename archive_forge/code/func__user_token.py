import keystoneauth1.exceptions as kc_exception
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
def _user_token(self):
    project_id = self.stack.stack_user_project_id
    if not project_id:
        raise ValueError(_("Can't get user token, user not yet created"))
    password = getattr(self, 'password', None)
    if password is None:
        raise ValueError(_("Can't get user token without password"))
    return self.keystone().stack_domain_user_token(user_id=self._get_user_id(), project_id=project_id, password=password)