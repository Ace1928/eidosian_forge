from neutron_lib.api.definitions import qos
from neutron_lib.api.definitions import qos_bw_minimum_ingress
from neutron_lib.services.qos import constants as qos_constants
from neutron_lib.tests.unit.api.definitions import base
class QoSBwMinimumIngressDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = qos_bw_minimum_ingress
    extension_subresources = (qos.MIN_BANDWIDTH_RULES,)
    extension_attributes = (qos_constants.DIRECTION,)