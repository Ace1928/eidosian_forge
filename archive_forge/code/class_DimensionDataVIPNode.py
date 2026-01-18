from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataVIPNode:

    def __init__(self, id, name, status, ip, connection_limit='10000', connection_rate_limit='10000'):
        """
        Initialize an instance of :class:`DimensionDataVIPNode`

        :param id: The ID of the node
        :type  id: ``str``

        :param name: The name of the node
        :type  name: ``str``

        :param status: The status of the node
        :type  status: :class:`DimensionDataStatus`

        :param ip: The IP of the node
        :type  ip: ``str``

        :param connection_limit: The total connection limit for the node
        :type  connection_limit: ``int``

        :param connection_rate_limit: The rate limit for the node
        :type  connection_rate_limit: ``int``
        """
        self.id = str(id)
        self.name = name
        self.status = status
        self.ip = ip
        self.connection_limit = connection_limit
        self.connection_rate_limit = connection_rate_limit

    def __repr__(self):
        return '<DimensionDataVIPNode: id=%s, name=%s, status=%s, ip=%s>' % (self.id, self.name, self.status, self.ip)