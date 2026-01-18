from heat.common import exception
from heat.engine import constraints
class KeystoneProjectConstraint(KeystoneBaseConstraint):
    resource_getter_name = 'get_project_id'
    entity = 'KeystoneProject'