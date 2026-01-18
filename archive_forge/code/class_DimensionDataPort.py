from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataPort:
    """
    A representation of Port in Dimension Data
    """

    def __init__(self, begin, end=None):
        """
        Initialize an instance of :class:`DimensionDataPort`

        :param begin: Port Number Begin
        :type  begin: ``str``

        :param end: Port Number end
        :type  end: ``str``
        """
        self.begin = begin
        self.end = end

    def __repr__(self):
        return '<DimensionDataPort: begin={}, end={}>'.format(self.begin, self.end)