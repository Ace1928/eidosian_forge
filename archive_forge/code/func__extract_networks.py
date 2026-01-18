import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
def _extract_networks(self, compute):
    """
        Extract networks from a compute node XML representation.

        Extract network descriptions from a compute node XML representation,
        converting each network to an OpenNebulaNetwork object.

        :type  compute: :class:`ElementTree`
        :param compute: XML representation of a compute node.

        :rtype:  ``list`` of :class:`OpenNebulaNetwork`
        :return: List of virtual networks attached to the compute node.
        """
    networks = []
    for element in compute.findall('NIC'):
        network = element.find('NETWORK')
        network_id = network.attrib['href'].partition('/network/')[2]
        networks.append(OpenNebulaNetwork(id=network_id, name=network.attrib.get('name', None), address=element.findtext('IP'), size=1, driver=self.connection.driver, extra={'mac': element.findtext('MAC')}))
    return networks