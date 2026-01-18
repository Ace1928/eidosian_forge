from openstack.database.v1 import _proxy
from openstack import service_description
class DatabaseService(service_description.ServiceDescription):
    """The database service."""
    supported_versions = {'1': _proxy.Proxy}