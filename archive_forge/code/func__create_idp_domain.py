import uuid
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.federation import utils
from keystone.i18n import _
from keystone import notifications
def _create_idp_domain(self, idp_id):
    domain_id = uuid.uuid4().hex
    desc = 'Auto generated federated domain for Identity Provider: '
    desc += idp_id
    domain = {'id': domain_id, 'name': domain_id, 'description': desc, 'enabled': True}
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    return domain_id