from openstack.block_storage.v2 import _proxy as _v2_proxy
from openstack.block_storage.v3 import _proxy as _v3_proxy
from openstack import service_description
class BlockStorageService(service_description.ServiceDescription):
    """The block storage service."""
    supported_versions = {'3': _v3_proxy.Proxy, '2': _v2_proxy.Proxy}