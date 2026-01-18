import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackNode(Node):
    """
    Subclass of Node so we can expose our extension methods.
    """

    def ex_allocate_public_ip(self):
        """
        Allocate a public IP and bind it to this node.
        """
        return self.driver.ex_allocate_public_ip(self)

    def ex_release_public_ip(self, address):
        """
        Release a public IP that this node holds.
        """
        return self.driver.ex_release_public_ip(self, address)

    def ex_create_ip_forwarding_rule(self, address, protocol, start_port, end_port=None):
        """
        Add a NAT/firewall forwarding rule for a port or ports.
        """
        return self.driver.ex_create_ip_forwarding_rule(node=self, address=address, protocol=protocol, start_port=start_port, end_port=end_port)

    def ex_create_port_forwarding_rule(self, address, private_port, public_port, protocol, public_end_port=None, private_end_port=None, openfirewall=True):
        """
        Add a port forwarding rule for port or ports.
        """
        return self.driver.ex_create_port_forwarding_rule(node=self, address=address, private_port=private_port, public_port=public_port, protocol=protocol, public_end_port=public_end_port, private_end_port=private_end_port, openfirewall=openfirewall)

    def ex_delete_ip_forwarding_rule(self, rule):
        """
        Delete a port forwarding rule.
        """
        return self.driver.ex_delete_ip_forwarding_rule(node=self, rule=rule)

    def ex_delete_port_forwarding_rule(self, rule):
        """
        Delete a NAT/firewall rule.
        """
        return self.driver.ex_delete_port_forwarding_rule(node=self, rule=rule)

    def ex_restore(self, template=None):
        """
        Restore virtual machine
        """
        return self.driver.ex_restore(node=self, template=template)

    def ex_change_node_size(self, offering):
        """
        Change virtual machine offering/size
        """
        return self.driver.ex_change_node_size(node=self, offering=offering)

    def ex_start(self):
        """
        Starts a stopped virtual machine.
        """
        return self.driver.ex_start(node=self)

    def ex_stop(self):
        """
        Stops a running virtual machine.
        """
        return self.driver.ex_stop(node=self)