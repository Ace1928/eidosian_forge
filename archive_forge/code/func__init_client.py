import json
from oslo_log import log as logging
from oslo_utils import importutils
from oslo_utils import uuidutils
from zaqarclient.transport import base
from zaqarclient.transport import request
from zaqarclient.transport import response
def _init_client(self, endpoint):
    """Initialize a websocket transport client.

        :param endpoint: The websocket endpoint. Example: ws://127.0.0.1:9000/.
                         Required.
        :type endpoint: string
        """
    self._websocket_client_id = uuidutils.generate_uuid()
    LOG.debug('Instantiating messaging websocket client: %s', endpoint)
    self._ws = self._create_connection(endpoint)
    auth_req = request.Request(endpoint, 'authenticate', headers={'X-Auth-Token': self._token})
    self.send(auth_req)