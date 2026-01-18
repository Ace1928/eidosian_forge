from neutron_lib.api.definitions import qos as qos_apidef
from neutron_lib.api.definitions import qos_pps_minimum_rule
from neutron_lib.services.qos import constants as qos_constants
from neutron_lib.tests.unit.api.definitions import base
class QoSPPSMinimumRuleDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = qos_pps_minimum_rule
    extension_resources = (qos_apidef.POLICIES,)
    extension_subresources = (qos_pps_minimum_rule.COLLECTION_NAME,)
    extension_attributes = (qos_constants.MIN_KPPS, qos_constants.DIRECTION)