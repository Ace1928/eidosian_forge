from neutron_lib.api.definitions import data_plane_status as dps
from neutron_lib.tests.unit.api.definitions import base
class DataPlaneStatusDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = dps
    extension_resources = (dps.COLLECTION_NAME,)
    extension_attributes = (dps.DATA_PLANE_STATUS,)