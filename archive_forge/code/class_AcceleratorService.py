from openstack.accelerator.v2 import _proxy as _proxy_v2
from openstack import service_description
class AcceleratorService(service_description.ServiceDescription):
    """The accelerator service."""
    supported_versions = {'2': _proxy_v2.Proxy}