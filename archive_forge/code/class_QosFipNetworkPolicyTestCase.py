from neutron_lib.api.definitions import l3 as l3_apidef
from neutron_lib.api.definitions import qos_fip_network_policy
from neutron_lib.services.qos import constants as qos_const
from neutron_lib.tests.unit.api.definitions import base
class QosFipNetworkPolicyTestCase(base.DefinitionBaseTestCase):
    extension_module = qos_fip_network_policy
    extension_resources = (l3_apidef.FLOATINGIPS,)
    extension_attributes = (qos_const.QOS_NETWORK_POLICY_ID,)