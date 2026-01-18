from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from pycadf import reason
from pycadf import resource
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _assert_domain_is_enabled(self, domain_ref):
    try:
        PROVIDERS.resource_api.assert_domain_enabled(domain_id=domain_ref['id'], domain=domain_ref)
    except AssertionError as e:
        LOG.warning(e)
        raise exception.Unauthorized from e