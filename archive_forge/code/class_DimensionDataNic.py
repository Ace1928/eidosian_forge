from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataNic:
    """
    A representation of Network Adapter in Dimension Data
    """

    def __init__(self, private_ip_v4=None, vlan=None, network_adapter_name=None):
        """
        Initialize an instance of :class:`DimensionDataNic`

        :param private_ip_v4: IPv4
        :type  private_ip_v4: ``str``

        :param vlan: Network VLAN
        :type  vlan: class: DimensionDataVlan or ``str``

        :param network_adapter_name: Network Adapter Name
        :type  network_adapter_name: ``str``
        """
        self.private_ip_v4 = private_ip_v4
        self.vlan = vlan
        self.network_adapter_name = network_adapter_name

    def __repr__(self):
        return '<DimensionDataNic: private_ip_v4=%s, vlan=%s,network_adapter_name=%s>' % (self.private_ip_v4, self.vlan, self.network_adapter_name)