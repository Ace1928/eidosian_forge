import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _create_node_mcp1(self, name, image, auth, ex_description, ex_network=None, ex_memory_gb=None, ex_cpu_specification=None, ex_is_started=True, ex_primary_dns=None, ex_secondary_dns=None, **kwargs):
    """
        Create a new NTTCIS node

        :keyword    name:   String with a name for this new node (required)
        :type       name:   ``str``

        :keyword    image:  OS Image to boot on node. (required)
        :type       image:  :class:`NodeImage` or ``str``

        :keyword    auth:   Initial authentication information for the
                            node. (If this is a customer LINUX
                            image auth will be ignored)
        :type       auth:   :class:`NodeAuthPassword` or ``str`` or
                            ``None``

        :keyword    ex_description:  description for this node (required)
        :type       ex_description:  ``str``

        :keyword    ex_network:  Network to create the node within
                                 (required unless using ex_network_domain
                                 or ex_primary_ipv4)

        :type       ex_network: :class:`NttCisNetwork` or ``str``

        :keyword    ex_memory_gb:  The amount of memory in GB for the
                                   server
        :type       ex_memory_gb: ``int``

        :keyword    ex_cpu_specification: The spec of CPU to deploy (
                                          optional)
        :type       ex_cpu_specification:
                        :class:`DimensionDataServerCpuSpecification`

        :keyword    ex_is_started:  Start server after creation? default
                                    true (required)
        :type       ex_is_started:  ``bool``

        :keyword    ex_primary_dns: The node's primary DNS

        :type       ex_primary_dns: ``str``

        :keyword    ex_secondary_dns: The node's secondary DNS

        :type       ex_secondary_dns: ``str``

        :return: The newly created :class:`Node`.
        :rtype: :class:`Node`
        """
    password = None
    image_needs_auth = self._image_needs_auth(image)
    if image_needs_auth:
        if isinstance(auth, basestring):
            auth_obj = NodeAuthPassword(password=auth)
            password = auth
        else:
            auth_obj = self._get_and_check_auth(auth)
            password = auth_obj.password
    server_elm = ET.Element('deployServer', {'xmlns': TYPES_URN})
    ET.SubElement(server_elm, 'name').text = name
    ET.SubElement(server_elm, 'description').text = ex_description
    image_id = self._image_to_image_id(image)
    ET.SubElement(server_elm, 'imageId').text = image_id
    ET.SubElement(server_elm, 'start').text = str(ex_is_started).lower()
    if password is not None:
        ET.SubElement(server_elm, 'administratorPassword').text = password
    if ex_cpu_specification is not None:
        cpu = ET.SubElement(server_elm, 'cpu')
        cpu.set('speed', ex_cpu_specification.performance)
        cpu.set('count', str(ex_cpu_specification.cpu_count))
        cpu.set('coresPerSocket', str(ex_cpu_specification.cores_per_socket))
    if ex_memory_gb is not None:
        ET.SubElement(server_elm, 'memoryGb').text = str(ex_memory_gb)
    if ex_network is not None:
        network_elm = ET.SubElement(server_elm, 'network')
        network_id = self._network_to_network_id(ex_network)
        ET.SubElement(network_elm, 'networkId').text = network_id
    if ex_primary_dns:
        dns_elm = ET.SubElement(server_elm, 'primaryDns')
        dns_elm.text = ex_primary_dns
    if ex_secondary_dns:
        dns_elm = ET.SubElement(server_elm, 'secondaryDns')
        dns_elm.text = ex_secondary_dns
    response = self.connection.request_with_orgId_api_2('server/deployServer', method='POST', data=ET.tostring(server_elm)).object
    node_id = None
    for info in findall(response, 'info', TYPES_URN):
        if info.get('name') == 'serverId':
            node_id = info.get('value')
    node = self.ex_get_node_by_id(node_id)
    if image_needs_auth:
        if getattr(auth_obj, 'generated', False):
            node.extra['password'] = auth_obj.password
    return node