from heat.common import exception
from heat.engine import constraints
class KeystoneDomainConstraint(KeystoneBaseConstraint):
    resource_getter_name = 'get_domain_id'
    entity = 'KeystoneDomain'