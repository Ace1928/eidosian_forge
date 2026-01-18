import abc
import keystone.conf
from keystone import exception
def _validate_default_domain_id(self, domain_id):
    """Validate that the domain ID belongs to the default domain."""
    if domain_id != CONF.identity.default_domain_id:
        raise exception.DomainNotFound(domain_id=domain_id)