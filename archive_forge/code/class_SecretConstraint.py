from barbicanclient import exceptions
from barbicanclient.v1 import client as barbican_client
from barbicanclient.v1 import containers
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
class SecretConstraint(constraints.BaseCustomConstraint):
    resource_client_name = CLIENT_NAME
    resource_getter_name = 'get_secret_by_ref'
    expected_exceptions = (exception.EntityNotFound,)