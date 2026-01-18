from openstack.container_infrastructure_management.v1 import _proxy
from openstack import service_description
class ContainerInfrastructureManagementService(service_description.ServiceDescription):
    """The container infrastructure management service."""
    supported_versions = {'1': _proxy.Proxy}