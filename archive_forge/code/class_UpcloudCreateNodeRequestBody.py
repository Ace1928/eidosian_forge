import json
import time
from libcloud.common.types import LibcloudError
from libcloud.common.exceptions import BaseHTTPError
class UpcloudCreateNodeRequestBody:
    """
    Body of the create_node request

    Takes the create_node arguments (**kwargs) and constructs the request body

    :param      name: Name of the created server (required)
    :type       name: ``str``

    :param      size: The size of resources allocated to this node.
    :type       size: :class:`.NodeSize`

    :param      image: OS Image to boot on node.
    :type       image: :class:`.NodeImage`

    :param      location: Which data center to create a node in. If empty,
                        undefined behavior will be selected. (optional)
    :type       location: :class:`.NodeLocation`

    :param      auth: Initial authentication information for the node
                            (optional)
    :type       auth: :class:`.NodeAuthSSHKey`

    :param      ex_hostname: Hostname. Default is 'localhost'. (optional)
    :type       ex_hostname: ``str``

    :param ex_username: User's username, which is created.
                        Default is 'root'. (optional)
    :type ex_username: ``str``
    """

    def __init__(self, name, size, image, location, auth=None, ex_hostname='localhost', ex_username='root'):
        self.body = {'server': {'title': name, 'hostname': ex_hostname, 'plan': size.id, 'zone': location.id, 'login_user': _LoginUser(ex_username, auth).to_dict(), 'storage_devices': _StorageDevice(image, size).to_dict()}}

    def to_json(self):
        """
        Serializes the body to json

        :return: JSON string
        :rtype: ``str``
        """
        return json.dumps(self.body)