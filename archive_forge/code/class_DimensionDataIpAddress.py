from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataIpAddress:
    """
    A representation of IP Address in Dimension Data
    """

    def __init__(self, begin, end=None, prefix_size=None):
        """
        Initialize an instance of :class:`DimensionDataIpAddress`

        :param begin: IP Address Begin
        :type  begin: ``str``

        :param end: IP Address end
        :type  end: ``str``

        :param prefixSize: IP Address prefix size
        :type  prefixSize: ``int``
        """
        self.begin = begin
        self.end = end
        self.prefix_size = prefix_size

    def __repr__(self):
        return '<DimensionDataIpAddress: begin={}, end={}, prefix_size={}>'.format(self.begin, self.end, self.prefix_size)