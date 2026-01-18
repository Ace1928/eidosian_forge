from openstack.clustering.v1 import _proxy
from openstack import service_description
class ClusteringService(service_description.ServiceDescription):
    """The clustering service."""
    supported_versions = {'1': _proxy.Proxy}