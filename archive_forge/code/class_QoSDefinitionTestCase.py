from neutron_lib.api.definitions import qos
from neutron_lib.services.qos import constants as q_const
from neutron_lib.tests.unit.api.definitions import base
class QoSDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = qos
    extension_resources = (qos.POLICIES, qos.RULE_TYPES)
    extension_subresources = (qos.BANDWIDTH_LIMIT_RULES, qos.DSCP_MARKING_RULES, qos.MIN_BANDWIDTH_RULES)
    extension_attributes = (q_const.DIRECTION, q_const.MAX_BURST, 'type', q_const.DSCP_MARK, q_const.MIN_KBPS, 'rules', q_const.MAX_KBPS, q_const.QOS_POLICY_ID)