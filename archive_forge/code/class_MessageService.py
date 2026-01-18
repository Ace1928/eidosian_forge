from openstack.message.v2 import _proxy
from openstack import service_description
class MessageService(service_description.ServiceDescription):
    """The message service."""
    supported_versions = {'2': _proxy.Proxy}