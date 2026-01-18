import functools
import typing as ty
import urllib
from urllib.parse import urlparse
import iso8601
import jmespath
from keystoneauth1 import adapter
from openstack import _log
from openstack import exceptions
from openstack import resource
def _bulk_create(self, resource_type: ty.Type[ResourceType], data, base_path=None) -> ty.Generator[ResourceType, None, None]:
    """Create a resource from attributes

        :param resource_type: The type of resource to create.
        :type resource_type: :class:`~openstack.resource.Resource`
        :param list data: List of attributes dicts to be passed onto the
            :meth:`~openstack.resource.Resource.create`
            method to be created. These should correspond
            to either :class:`~openstack.resource.Body`
            or :class:`~openstack.resource.Header`
            values on this resource.
        :param str base_path: Base part of the URI for creating resources, if
            different from
            :data:`~openstack.resource.Resource.base_path`.

        :returns: A generator of Resource objects.
        :rtype: :class:`~openstack.resource.Resource`
        """
    return resource_type.bulk_create(self, data, base_path=base_path)