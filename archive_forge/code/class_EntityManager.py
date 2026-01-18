import abc
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneclient import base
class EntityManager(base.Manager, metaclass=abc.ABCMeta):
    """Manager class for listing federated accessible objects."""
    resource_class = None

    @property
    @abc.abstractmethod
    def object_type(self):
        raise exceptions.MethodNotImplemented

    def list(self):
        url = '/auth/%s' % self.object_type
        try:
            tenant_list = self._list(url, self.object_type)
        except exceptions.CatalogException:
            endpoint_filter = {'interface': plugin.AUTH_INTERFACE}
            tenant_list = self._list(url, self.object_type, endpoint_filter=endpoint_filter)
        return tenant_list