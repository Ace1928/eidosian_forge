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
def _only_allow_enabled_to_update_cascade(self, project, original_project):
    for attr in project:
        if attr != 'enabled':
            if project.get(attr) != original_project.get(attr):
                raise exception.ValidationError(message=_('Cascade update is only allowed for enabled attribute.'))