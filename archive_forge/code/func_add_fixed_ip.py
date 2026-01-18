import typing as ty
from openstack.common import metadata
from openstack.common import tag
from openstack.compute.v2 import flavor
from openstack.compute.v2 import volume_attachment
from openstack import exceptions
from openstack.image.v2 import image
from openstack import resource
from openstack import utils
def add_fixed_ip(self, session, network_id):
    """Add a fixed IP to the server.

        This is effectively an alias for adding a network.

        :param session: The session to use for making this request.
        :param network_id: The network to connect the server to.
        :returns: None
        """
    body = {'addFixedIp': {'networkId': network_id}}
    self._action(session, body)