import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def add_floating_ip(self, session, address, fixed_address=None):
    """Add a floating IP to the server.

        :param session: The session to use for making this request.
        :param address: The floating IP address to associate with the server.
        :param fixed_address: A fixed IP address with which to associated the
            floating IP. (Optional)
        :returns: None
        """
    body = {'addFloatingIp': {'address': address}}
    if fixed_address is not None:
        body['addFloatingIp']['fixed_address'] = fixed_address
    self._action(session, body)