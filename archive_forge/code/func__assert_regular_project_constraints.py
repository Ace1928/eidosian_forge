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
def _assert_regular_project_constraints(self, project_ref):
    """Enforce regular project hierarchy constraints.

        Called when is_domain is false. The project must contain a valid
        domain_id and parent_id. The goal of this method is to check
        that the domain_id specified is consistent with the domain of its
        parent.

        :raises keystone.exception.ValidationError: If one of the constraints
            was not satisfied.
        :raises keystone.exception.DomainNotFound: In case the domain is not
            found.
        """
    domain = self.get_domain(project_ref['domain_id'])
    parent_ref = self.get_project(project_ref['parent_id'])
    if parent_ref['is_domain']:
        if parent_ref['id'] != domain['id']:
            raise exception.ValidationError(message=_("Cannot create project, the parent (%(parent_id)s) is acting as a domain, but this project's domain id (%(domain_id)s) does not match the parent's id.") % {'parent_id': parent_ref['id'], 'domain_id': domain['id']})
    else:
        parent_domain_id = parent_ref.get('domain_id')
        if parent_domain_id != domain['id']:
            raise exception.ValidationError(message=_('Cannot create project, since it specifies its domain_id %(domain_id)s, but specifies a parent in a different domain (%(parent_domain_id)s).') % {'domain_id': domain['id'], 'parent_domain_id': parent_domain_id})