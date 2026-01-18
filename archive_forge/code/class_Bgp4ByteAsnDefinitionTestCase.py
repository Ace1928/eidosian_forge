from neutron_lib.api.definitions import bgp_4byte_asn
from neutron_lib.tests.unit.api.definitions import base
class Bgp4ByteAsnDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = bgp_4byte_asn
    extension_resources = ('bgp-speakers', 'bgp-peers')
    extension_attributes = ('local_as', 'remote_as')