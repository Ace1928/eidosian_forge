from neutron_lib.api.definitions import agent
from neutron_lib.api.definitions import availability_zone
from neutron_lib.tests.unit.api.definitions import base
class AvailabilityZoneDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = availability_zone
    extension_resources = (availability_zone.COLLECTION_NAME, agent.COLLECTION_NAME)
    extension_attributes = (availability_zone.AZ_HINTS, 'resource', availability_zone.RESOURCE_NAME, 'state')