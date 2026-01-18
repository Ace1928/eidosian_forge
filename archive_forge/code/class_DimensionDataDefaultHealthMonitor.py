from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataDefaultHealthMonitor:
    """
    A default health monitor for a VIP (node, pool or listener)
    """

    def __init__(self, id, name, node_compatible, pool_compatible):
        """
        Initialize an instance of :class:`DimensionDataDefaultHealthMonitor`

        :param id: The ID of the monitor
        :type  id: ``str``

        :param name: The name of the monitor
        :type  name: ``str``

        :param node_compatible: Is a monitor capable of monitoring nodes
        :type  node_compatible: ``bool``

        :param pool_compatible: Is a monitor capable of monitoring pools
        :type  pool_compatible: ``bool``
        """
        self.id = id
        self.name = name
        self.node_compatible = node_compatible
        self.pool_compatible = pool_compatible

    def __repr__(self):
        return '<DimensionDataDefaultHealthMonitor: id=%s, name=%s>' % (self.id, self.name)