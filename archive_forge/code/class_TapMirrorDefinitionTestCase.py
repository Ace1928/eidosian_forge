from neutron_lib.api.definitions import tap_mirror
from neutron_lib.tests.unit.api.definitions import base
class TapMirrorDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = tap_mirror
    extension_resources = (tap_mirror.COLLECTION_NAME,)
    extension_attributes = ('port_id', 'remote_ip', 'directions', 'mirror_type')