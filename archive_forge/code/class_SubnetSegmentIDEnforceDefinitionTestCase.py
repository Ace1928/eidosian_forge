from neutron_lib.api.definitions import segment
from neutron_lib.api.definitions import subnet_segmentid_enforce
from neutron_lib.tests.unit.api.definitions import base
class SubnetSegmentIDEnforceDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = subnet_segmentid_enforce
    extension_attributes = (segment.SEGMENT_ID,)