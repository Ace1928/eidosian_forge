from neutron_lib.api.definitions import local_ip
from neutron_lib.tests.unit.api.definitions import base
class LocalIPDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = local_ip
    extension_resources = (local_ip.COLLECTION_NAME,)
    extension_subresources = (local_ip.LOCAL_IP_ASSOCIATIONS,)
    extension_attributes = ('local_port_id', 'local_ip_address', 'ip_mode', 'local_ip_id', 'fixed_port_id', 'fixed_ip', 'host')