import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
class OpenStack_2_PortInterface(UuidMixin):
    """
    Port Interface info. Similar in functionality to a floating IP (can be
    attached / detached from a compute instance) but implementation-wise a
    bit different.

    > A port is a connection point for attaching a single device, such as the
    > NIC of a server, to a network. The port also describes the associated
    > network configuration, such as the MAC and IP addresses to be used on
    > that port.
    https://docs.openstack.org/python-openstackclient/pike/cli/command-objects/port.html

    Also see:
    https://developer.openstack.org/api-ref/compute/#port-interfaces-servers-os-interface
    """

    def __init__(self, id, state, driver, created=None, extra=None):
        """
        :param id: Port Interface ID.
        :type id: ``str``
        :param state: State of the OpenStack_2_PortInterface.
        :type state: :class:`.OpenStack_2_PortInterfaceState`
        :param      created: A datetime object that represents when the
                             port interface was created
        :type       created: ``datetime.datetime``
        :param extra: Optional provided specific attributes associated with
                      this image.
        :type extra: ``dict``
        """
        self.id = str(id)
        self.state = state
        self.driver = driver
        self.created = created
        self.extra = extra or {}
        UuidMixin.__init__(self)

    def delete(self):
        """
        Delete this Port Interface

        :rtype: ``bool``
        """
        return self.driver.ex_delete_port(self)

    def __repr__(self):
        return '<OpenStack_2_PortInterface: id=%s, state=%s, driver=%s  ...>' % (self.id, self.state, self.driver.name)