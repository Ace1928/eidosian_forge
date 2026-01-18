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
def _get_uri_attribute(self, child, parent, name):
    """Get a value to be associated with a URI attribute

        `child` will not be None here as it's a required argument
        on the proxy method. `parent` is allowed to be None if `child`
        is an actual resource, but when an ID is given for the child
        one must also be provided for the parent. An example of this
        is that a parent is a Server and a child is a ServerInterface.
        """
    if parent is None:
        value = getattr(child, name)
    else:
        value = resource.Resource._get_id(parent)
    return value