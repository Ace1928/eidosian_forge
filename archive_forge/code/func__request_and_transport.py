from oslo_utils import uuidutils
from zaqarclient.common import decorators
from zaqarclient.queues.v1 import core
from zaqarclient.queues.v1 import flavor
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import pool
from zaqarclient.queues.v1 import queues
from zaqarclient import transport
from zaqarclient.transport import errors
from zaqarclient.transport import request
def _request_and_transport(self):
    req = request.prepare_request(self.auth_opts, endpoint=self.api_url, api=self.api_version, session=self.session)
    req.headers['Client-ID'] = self.client_uuid
    trans = self._get_transport(req)
    return (req, trans)