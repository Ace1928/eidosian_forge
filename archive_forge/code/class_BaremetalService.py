from openstack.baremetal.v1 import _proxy
from openstack import service_description
class BaremetalService(service_description.ServiceDescription):
    """The bare metal service."""
    supported_versions = {'1': _proxy.Proxy}