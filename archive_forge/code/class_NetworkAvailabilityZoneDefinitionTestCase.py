from neutron_lib.api.definitions import availability_zone
from neutron_lib.api.definitions import network_availability_zone
from neutron_lib.tests.unit.api.definitions import base
class NetworkAvailabilityZoneDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = network_availability_zone
    extension_attributes = (availability_zone.AZ_HINTS, availability_zone.COLLECTION_NAME)