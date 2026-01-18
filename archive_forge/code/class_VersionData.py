import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
class VersionData(dict):
    """Normalized Version Data about an endpoint."""

    def __init__(self, version, url, collection=None, max_microversion=None, min_microversion=None, next_min_version=None, not_before=None, status='CURRENT', raw_status=None):
        super(VersionData, self).__init__()
        self['version'] = version
        self['url'] = url
        self['collection'] = collection
        self['max_microversion'] = max_microversion
        self['min_microversion'] = min_microversion
        self['next_min_version'] = next_min_version
        self['not_before'] = not_before
        self['status'] = status
        self['raw_status'] = raw_status

    @property
    def version(self):
        """The normalized version of the endpoint."""
        return self.get('version')

    @property
    def url(self):
        """The url for the endpoint."""
        return self.get('url')

    @property
    def collection(self):
        """The URL for the discovery document.

        May be None.
        """
        return self.get('collection')

    @property
    def min_microversion(self):
        """The minimum microversion supported by the endpoint.

        May be None.
        """
        return self.get('min_microversion')

    @property
    def max_microversion(self):
        """The maximum microversion supported by the endpoint.

        May be None.
        """
        return self.get('max_microversion')

    @property
    def status(self):
        """A canonicalized version of the status.

        Valid values are CURRENT, SUPPORTED, DEPRECATED and EXPERIMENTAL.
        """
        return self.get('status')

    @property
    def raw_status(self):
        """The status as provided by the server."""
        return self.get('raw_status')