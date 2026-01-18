from designateclient import client
from designateclient import exceptions
from heat.common import exception as heat_exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
class DesignateZoneConstraint(constraints.BaseCustomConstraint):
    resource_client_name = CLIENT_NAME
    resource_getter_name = 'get_zone_id'