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
def _delete_domain(self, domain, initiator=None):
    ro_opt.check_immutable_delete(resource_ref=domain, resource_type='domain', resource_id=domain['id'])
    if domain['enabled']:
        raise exception.ForbiddenNotSecurity(_('Cannot delete a domain that is enabled, please disable it first.'))
    domain_id = domain['id']
    self._delete_domain_contents(domain_id)
    notifications.Audit.internal(notifications.DOMAIN_DELETED, domain_id)
    self._delete_project(domain, initiator)
    try:
        self.get_domain.invalidate(self, domain_id)
        self.get_domain_by_name.invalidate(self, domain['name'])
        PROVIDERS.domain_config_api.delete_config_options(domain_id)
        PROVIDERS.domain_config_api.release_registration(domain_id)
    finally:
        notifications.Audit.deleted(self._DOMAIN, domain_id, initiator)