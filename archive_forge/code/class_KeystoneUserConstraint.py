from heat.common import exception
from heat.engine import constraints
class KeystoneUserConstraint(KeystoneBaseConstraint):
    resource_getter_name = 'get_user_id'
    entity = 'KeystoneUser'