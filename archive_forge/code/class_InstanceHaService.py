from openstack.instance_ha.v1 import _proxy
from openstack import service_description
class InstanceHaService(service_description.ServiceDescription):
    """The HA service."""
    supported_versions = {'1': _proxy.Proxy}