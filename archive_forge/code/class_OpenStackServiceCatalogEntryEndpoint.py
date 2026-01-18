import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackServiceCatalogEntryEndpoint:
    VALID_ENDPOINT_TYPES = [OpenStackIdentityEndpointType.INTERNAL, OpenStackIdentityEndpointType.EXTERNAL, OpenStackIdentityEndpointType.ADMIN]

    def __init__(self, region, url, endpoint_type='external'):
        """
        :param region: Endpoint region.
        :type region: ``str``

        :param url: Endpoint URL.
        :type url: ``str``

        :param endpoint_type: Endpoint type (external / internal / admin).
        :type endpoint_type: ``str``
        """
        if endpoint_type not in self.VALID_ENDPOINT_TYPES:
            raise ValueError('Invalid type: %s' % endpoint_type)
        self.region = region
        self.url = url
        self.endpoint_type = endpoint_type

    def __eq__(self, other):
        return self.region == other.region and self.url == other.url and (self.endpoint_type == other.endpoint_type)

    def __ne__(self, other):
        return not self.__eq__(other=other)

    def __repr__(self):
        return '<OpenStackServiceCatalogEntryEndpoint region=%s, url=%s, type=%s' % (self.region, self.url, self.endpoint_type)