from neutron_lib.api.definitions import sfc
from neutron_lib.tests.unit.api.definitions import base
class SFCDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = sfc
    extension_resources = tuple(sfc.RESOURCE_ATTRIBUTE_MAP.keys())
    extension_attributes = ('type', 'ingress', 'egress', 'service_function_parameters', 'group_id', 'port_pairs', 'port_pair_group_parameters', 'chain_id', 'port_pair_groups', 'flow_classifiers', 'chain_parameters')