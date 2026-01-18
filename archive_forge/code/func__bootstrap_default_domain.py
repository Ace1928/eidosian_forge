import uuid
from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.server import backends
def _bootstrap_default_domain(self):
    default_domain = {'id': CONF.identity.default_domain_id, 'name': 'Default', 'enabled': True, 'description': 'The default domain'}
    try:
        PROVIDERS.resource_api.create_domain(domain_id=default_domain['id'], domain=default_domain)
        LOG.info('Created domain %s', default_domain['id'])
    except exception.Conflict:
        LOG.info('Domain %s already exists, skipping creation.', default_domain['id'])
    self.default_domain_id = default_domain['id']