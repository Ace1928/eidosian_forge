from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataNetwork:
    """
    DimensionData network with location.
    """

    def __init__(self, id, name, description, location, private_net, multicast, status):
        self.id = str(id)
        self.name = name
        self.description = description
        self.location = location
        self.private_net = private_net
        self.multicast = multicast
        self.status = status

    def __repr__(self):
        return '<DimensionDataNetwork: id=%s, name=%s, description=%s, location=%s, private_net=%s, multicast=%s>' % (self.id, self.name, self.description, self.location, self.private_net, self.multicast)