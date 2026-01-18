from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.exc
from glance.api import policy
from glance.common import wsgi
import glance.context
from glance.i18n import _, _LW
def _get_anonymous_context(self):
    kwargs = {'user': None, 'tenant': None, 'roles': [], 'is_admin': False, 'read_only': True, 'policy_enforcer': self.policy_enforcer}
    return glance.context.RequestContext(**kwargs)