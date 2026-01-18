from openstack.common import tag
from openstack import exceptions
from openstack.network.v2 import _base
from openstack import resource
from openstack import utils
def add_gateway(self, session, **body):
    """Add an external gateway to a logical router.

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param dict body: The body requested to be updated on the router

        :returns: The body of the response as a dictionary.
        """
    url = utils.urljoin(self.base_path, self.id, 'add_gateway_router')
    resp = session.put(url, json=body)
    return resp.json()