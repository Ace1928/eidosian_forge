from neutron_lib.api.definitions import l3 as l3_apidef
from neutron_lib.api.definitions import qos_fip as qos_fip_apidef
from neutron_lib.tests.unit.api.definitions import base
class QoSFIPTestCase(base.DefinitionBaseTestCase):
    extension_module = qos_fip_apidef
    extension_resources = (l3_apidef.FLOATINGIPS,)