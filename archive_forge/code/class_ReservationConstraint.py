from blazarclient import client as blazar_client
from blazarclient import exception as client_exception
from oslo_config import cfg
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
class ReservationConstraint(BlazarBaseConstraint):
    expected_exceptions = (exception.EntityNotFound, client_exception.BlazarClientException)
    resource_getter_name = 'get_lease'