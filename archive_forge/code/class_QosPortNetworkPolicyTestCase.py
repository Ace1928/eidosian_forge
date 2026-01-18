from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import qos_port_network_policy
from neutron_lib.services.qos import constants as qos_const
from neutron_lib.tests.unit.api.definitions import base
class QosPortNetworkPolicyTestCase(base.DefinitionBaseTestCase):
    extension_module = qos_port_network_policy
    extension_resources = (port.RESOURCE_NAME,)
    extension_attributes = (qos_const.QOS_NETWORK_POLICY_ID,)