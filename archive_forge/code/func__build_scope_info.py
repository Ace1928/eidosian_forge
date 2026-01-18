import hashlib
from oslo_log import log
from keystone.auth import core
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils
from keystone.i18n import _
def _build_scope_info(self):
    """Build the token request scope based on the headers.

        :returns: scope data
        :rtype: dict
        """
    project_id = self.env.get('HTTP_X_PROJECT_ID')
    project_name = self.env.get('HTTP_X_PROJECT_NAME')
    project_domain_id = self.env.get('HTTP_X_PROJECT_DOMAIN_ID')
    project_domain_name = self.env.get('HTTP_X_PROJECT_DOMAIN_NAME')
    domain_id = self.env.get('HTTP_X_DOMAIN_ID')
    domain_name = self.env.get('HTTP_X_DOMAIN_NAME')
    scope = {}
    if project_id:
        scope['project'] = {'id': project_id}
    elif project_name:
        scope['project'] = {'name': project_name}
        if project_domain_id:
            scope['project']['domain'] = {'id': project_domain_id}
        elif project_domain_name:
            scope['project']['domain'] = {'name': project_domain_name}
        else:
            msg = _('Neither Project Domain ID nor Project Domain Name was provided.')
            raise exception.ValidationError(msg)
    elif domain_id:
        scope['domain'] = {'id': domain_id}
    elif domain_name:
        scope['domain'] = {'name': domain_name}
    else:
        raise exception.ValidationError(attribute='project or domain', target='scope')
    return scope