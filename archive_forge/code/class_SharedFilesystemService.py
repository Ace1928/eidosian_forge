from openstack import service_description
from openstack.shared_file_system.v2 import _proxy
class SharedFilesystemService(service_description.ServiceDescription):
    """The shared file systems service."""
    supported_versions = {'2': _proxy.Proxy}