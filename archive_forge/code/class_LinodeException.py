from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.gandi import BaseObject
from libcloud.common.types import LibcloudError, InvalidCredsError
class LinodeException(Exception):
    """Error originating from the Linode API

    This class wraps a Linode API error, a list of which is available in the
    API documentation.  All Linode API errors are a numeric code and a
    human-readable description.
    """

    def __init__(self, code, message):
        self.code = code
        self.message = message
        self.args = (code, message)

    def __str__(self):
        return '(%u) %s' % (self.code, self.message)

    def __repr__(self):
        return "<LinodeException code %u '%s'>" % (self.code, self.message)