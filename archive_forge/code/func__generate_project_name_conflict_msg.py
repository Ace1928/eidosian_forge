from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def _generate_project_name_conflict_msg(self, project):
    if project['is_domain']:
        return _('it is not permitted to have two projects acting as domains with the same name: %s') % project['name']
    else:
        return _('it is not permitted to have two projects with either the same name or same id in the same domain: name is %(name)s, project id %(id)s') % project