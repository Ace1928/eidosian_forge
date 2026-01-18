from heat.common import exception as heat_exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
from manilaclient import client as manila_client
from manilaclient import exceptions
from oslo_config import cfg
class ManilaShareBaseConstraint(constraints.BaseCustomConstraint):
    expected_exceptions = (heat_exception.EntityNotFound, exceptions.NoUniqueMatch)
    resource_client_name = CLIENT_NAME