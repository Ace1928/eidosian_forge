from neutron_lib.api.definitions import l3_ext_ndp_proxy
from neutron_lib.tests.unit.api.definitions import base
class RouterNDPProxyDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = l3_ext_ndp_proxy
    extension_attributes = ('enable_ndp_proxy',)