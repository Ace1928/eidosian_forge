from neutron_lib.api.definitions import l3
from neutron_lib.api.definitions import qos_gateway_ip
from neutron_lib.tests.unit.api.definitions import base
class QosGatewayIPDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = qos_gateway_ip
    extension_attributes = (l3.EXTERNAL_GW_INFO,)