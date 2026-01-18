from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataNetworkDomain:
    """
    DimensionData network domain with location.
    """

    def __init__(self, id, name, description, location, status, plan):
        self.id = str(id)
        self.name = name
        self.description = description
        self.location = location
        self.status = status
        self.plan = plan

    def __repr__(self):
        return '<DimensionDataNetworkDomain: id=%s, name=%s, description=%s, location=%s, status=%s, plan=%s>' % (self.id, self.name, self.description, self.location, self.status, self.plan)