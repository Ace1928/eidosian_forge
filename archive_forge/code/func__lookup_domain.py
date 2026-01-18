from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from pycadf import reason
from pycadf import resource
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _lookup_domain(self, domain_info):
    domain_id = domain_info.get('id')
    domain_name = domain_info.get('name')
    if not domain_id and (not domain_name):
        raise exception.ValidationError(attribute='id or name', target='domain')
    try:
        if domain_name:
            domain_ref = PROVIDERS.resource_api.get_domain_by_name(domain_name)
        else:
            domain_ref = PROVIDERS.resource_api.get_domain(domain_id)
    except exception.DomainNotFound as e:
        LOG.warning(e)
        raise exception.Unauthorized(e)
    self._assert_domain_is_enabled(domain_ref)
    return domain_ref