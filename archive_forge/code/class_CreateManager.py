import abc
import copy
from urllib import parse as urlparse
from ironicclient.common.apiclient import base
from ironicclient import exc
class CreateManager(Manager, metaclass=abc.ABCMeta):
    """Provides creation operations with a particular API."""

    @property
    @abc.abstractmethod
    def _creation_attributes(self):
        """A list of required creation attributes for a resource type.

        """

    def create(self, os_ironic_api_version=None, global_request_id=None, **kwargs):
        """Create a resource based on a kwargs dictionary of attributes.

        :param kwargs: A dictionary containing the attributes of the resource
                       that will be created.
        :param os_ironic_api_version: String version (e.g. "1.35") to use for
            the request.  If not specified, the client's default is used.
        :param global_request_id: String containing global request ID header
            value (in form "req-<UUID>") to use for the request.
        :raises exc.InvalidAttribute: For invalid attributes that are not
                                      needed to create the resource.
        """
        new = {}
        invalid = []
        for key, value in kwargs.items():
            if key in self._creation_attributes:
                new[key] = value
            else:
                invalid.append(key)
        if invalid:
            raise exc.InvalidAttribute('The attribute(s) "%(attrs)s" are invalid; they are not needed to create %(resource)s.' % {'resource': self._resource_name, 'attrs': '","'.join(invalid)})
        headers = {}
        if os_ironic_api_version is not None:
            headers['X-OpenStack-Ironic-API-Version'] = os_ironic_api_version
        if global_request_id is not None:
            headers['X-Openstack-Request-Id'] = global_request_id
        url = self._path()
        resp, body = self.api.json_request('POST', url, body=new, headers=headers)
        if body:
            return self.resource_class(self, body)