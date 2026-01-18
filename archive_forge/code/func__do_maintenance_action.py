import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def _do_maintenance_action(self, session, verb, body=None):
    session = self._get_session(session)
    version = self._get_microversion(session, action='commit')
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'maintenance')
    response = getattr(session, verb)(request.url, json=body, headers=request.headers, microversion=version)
    msg = 'Failed to change maintenance mode for node {node}'.format(node=self.id)
    exceptions.raise_from_response(response, error_message=msg)