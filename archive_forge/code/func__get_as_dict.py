import abc
import copy
from urllib import parse as urlparse
from ironicclient.common.apiclient import base
from ironicclient import exc
def _get_as_dict(self, resource_id, fields=None, os_ironic_api_version=None, global_request_id=None):
    """Retrieve a resource as a dictionary

        :param resource_id: Identifier of the resource.
        :param fields: List of specific fields to be returned.
        :param os_ironic_api_version: String version (e.g. "1.35") to use for
            the request.  If not specified, the client's default is used.
        :param global_request_id: String containing global request ID header
            value (in form "req-<UUID>") to use for the request.
        :returns: a dictionary representing the resource; may be empty
        """
    resource = self._get(resource_id, fields=fields, os_ironic_api_version=os_ironic_api_version, global_request_id=global_request_id)
    if resource:
        return resource.to_dict()
    else:
        return {}