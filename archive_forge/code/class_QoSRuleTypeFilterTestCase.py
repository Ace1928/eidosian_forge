from neutron_lib.api.definitions import qos_rule_type_filter
from neutron_lib.tests.unit.api.definitions import base
from neutron_lib.tests.unit.api.definitions import test_qos
class QoSRuleTypeFilterTestCase(base.DefinitionBaseTestCase):
    extension_module = qos_rule_type_filter
    extension_resources = test_qos.QoSDefinitionTestCase.extension_resources
    extension_attributes = (qos_rule_type_filter.QOS_RULE_TYPE_ALL_SUPPORTED, qos_rule_type_filter.QOS_RULE_TYPE_ALL_RULES)