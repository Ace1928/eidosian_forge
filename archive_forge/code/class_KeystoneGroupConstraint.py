from heat.common import exception
from heat.engine import constraints
class KeystoneGroupConstraint(KeystoneBaseConstraint):
    resource_getter_name = 'get_group_id'
    entity = 'KeystoneGroup'