from heat.common import exception
from heat.engine import constraints
class KeystoneServiceConstraint(KeystoneBaseConstraint):
    expected_exceptions = (exception.EntityNotFound, exception.KeystoneServiceNameConflict)
    resource_getter_name = 'get_service_id'
    entity = 'KeystoneService'