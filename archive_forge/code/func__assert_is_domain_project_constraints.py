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
def _assert_is_domain_project_constraints(self, project_ref):
    """Enforce specific constraints of projects that act as domains.

        Called when is_domain is true, this method ensures that:

        * multiple domains are enabled
        * the project name is not the reserved name for a federated domain
        * the project is a root project

        :raises keystone.exception.ValidationError: If one of the constraints
            was not satisfied.
        """
    if not PROVIDERS.identity_api.multiple_domains_supported and project_ref['id'] != CONF.identity.default_domain_id and (project_ref['id'] != base.NULL_DOMAIN_ID):
        raise exception.ValidationError(message=_('Multiple domains are not supported'))
    self.assert_domain_not_federated(project_ref['id'], project_ref)
    if project_ref['parent_id']:
        raise exception.ValidationError(message=_('only root projects are allowed to act as domains.'))