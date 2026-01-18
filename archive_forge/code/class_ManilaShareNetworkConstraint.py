from heat.common import exception as heat_exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
from manilaclient import client as manila_client
from manilaclient import exceptions
from oslo_config import cfg
class ManilaShareNetworkConstraint(ManilaShareBaseConstraint):
    resource_getter_name = 'get_share_network'