import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackIPForwardingRule:
    """
    A NAT/firewall forwarding rule.
    """

    def __init__(self, node, id, address, protocol, start_port, end_port=None):
        """
        A NAT/firewall forwarding rule.

        @note: This is a non-standard extension API, and only works for
               CloudStack.

        :param      node: Node for rule
        :type       node: :class:`Node`

        :param      id: Rule ID
        :type       id: ``int``

        :param      address: External IP address
        :type       address: :class:`CloudStackAddress`

        :param      protocol: TCP/IP Protocol (TCP, UDP)
        :type       protocol: ``str``

        :param      start_port: Start port for the rule
        :type       start_port: ``int``

        :param      end_port: End port for the rule
        :type       end_port: ``int``

        :rtype: :class:`CloudStackIPForwardingRule`
        """
        self.node = node
        self.id = id
        self.address = address
        self.protocol = protocol
        self.start_port = start_port
        self.end_port = end_port

    def delete(self):
        self.node.ex_delete_ip_forwarding_rule(rule=self)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.id == other.id