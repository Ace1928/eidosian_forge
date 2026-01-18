from neutron_lib.api.definitions import segment
from neutron_lib.api.definitions import subnet_segmentid_writable
from neutron_lib.tests.unit.api.definitions import base
class SubnetSegmentIDDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = subnet_segmentid_writable
    extension_attributes = (segment.SEGMENT_ID,)