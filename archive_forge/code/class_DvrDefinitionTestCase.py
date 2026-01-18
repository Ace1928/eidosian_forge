from neutron_lib.api.definitions import dvr
from neutron_lib.tests.unit.api.definitions import base
class DvrDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = dvr
    extension_resources = ()
    extension_attributes = (dvr.DISTRIBUTED,)