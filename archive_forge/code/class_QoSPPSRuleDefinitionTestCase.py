from neutron_lib.api.definitions import qos as qos_apidef
from neutron_lib.api.definitions import qos_pps_rule
from neutron_lib.services.qos import constants as qos_const
from neutron_lib.tests.unit.api.definitions import base
class QoSPPSRuleDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = qos_pps_rule
    extension_resources = (qos_apidef.POLICIES,)
    extension_subresources = (qos_apidef.BANDWIDTH_LIMIT_RULES, qos_apidef.DSCP_MARKING_RULES, qos_apidef.MIN_BANDWIDTH_RULES, qos_pps_rule.PACKET_RATE_LIMIT_RULES)
    extension_attributes = (qos_const.MAX_KPPS, qos_const.MAX_BURST_KPPS, qos_const.DIRECTION, qos_const.MAX_BURST, 'type', qos_const.DSCP_MARK, qos_const.MIN_KBPS, 'rules', qos_const.MAX_KBPS, qos_const.QOS_POLICY_ID)