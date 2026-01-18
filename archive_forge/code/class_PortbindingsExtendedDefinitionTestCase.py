from neutron_lib.api.definitions import portbindings_extended as pbe
from neutron_lib.tests.unit.api.definitions import base
class PortbindingsExtendedDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = pbe
    extension_attributes = (pbe.VIF_TYPE, pbe.VIF_DETAILS, pbe.VNIC_TYPE, pbe.HOST, pbe.PROFILE, pbe.STATUS, pbe.PROJECT_ID)
    extension_subresources = (pbe.COLLECTION_NAME,)