import json
import time
from libcloud.common.types import LibcloudError
from libcloud.common.exceptions import BaseHTTPError
class UpcloudNodeOperations:
    """
    Helper class to start and stop node.

    :param  connection: Connection instance
    :type   connection: :class:`.UpcloudConnection`
    """

    def __init__(self, connection):
        self.connection = connection

    def stop_node(self, node_id):
        """
        Stops the node

        :param  node_id: Id of the Node
        :type   node_id: ``int``
        """
        body = {'stop_server': {'stop_type': 'hard'}}
        self.connection.request('1.2/server/{}/stop'.format(node_id), method='POST', data=json.dumps(body))

    def get_node_state(self, node_id):
        """
        Get the state of the node.

        :param  node_id: Id of the Node
        :type   node_id: ``int``

        :rtype: ``str``
        """
        action = '1.2/server/{}'.format(node_id)
        try:
            response = self.connection.request(action)
            return response.object['server']['state']
        except BaseHTTPError as e:
            if e.code == 404:
                return None
            raise

    def destroy_node(self, node_id):
        """
        Destroys the node.

        :param  node_id: Id of the Node
        :type   node_id: ``int``
        """
        self.connection.request('1.2/server/{}'.format(node_id), method='DELETE')