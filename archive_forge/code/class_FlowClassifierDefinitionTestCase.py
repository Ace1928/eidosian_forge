from neutron_lib.api.definitions import flowclassifier
from neutron_lib.tests.unit.api.definitions import base
class FlowClassifierDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = flowclassifier
    extension_resources = (flowclassifier.COLLECTION_NAME,)
    extension_attributes = ('ethertype', 'protocol', 'source_port_range_min', 'source_port_range_max', 'destination_port_range_min', 'destination_port_range_max', 'source_ip_prefix', 'destination_ip_prefix', 'logical_destination_port', 'logical_source_port', 'l7_parameters')