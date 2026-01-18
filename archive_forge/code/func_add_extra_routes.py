from openstack.common import tag
from openstack import exceptions
from openstack.network.v2 import _base
from openstack import resource
from openstack import utils
def add_extra_routes(self, session, body) -> 'Router':
    """Add extra routes to a logical router.

        :param session: The session to communicate through.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param dict body: The request body as documented in the api-ref.

        :returns: The response as a Router object with the added extra routes.

        :raises: :class:`~openstack.exceptions.SDKException` on error.
        """
    url = utils.urljoin(self.base_path, self.id, 'add_extraroutes')
    resp = self._put(session, url, body)
    self._translate_response(resp)
    return self