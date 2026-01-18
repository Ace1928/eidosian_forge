from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataFirewallRule:
    """
    DimensionData Firewall Rule for a network domain
    """

    def __init__(self, id, name, action, location, network_domain, status, ip_version, protocol, source, destination, enabled):
        self.id = str(id)
        self.name = name
        self.action = action
        self.location = location
        self.network_domain = network_domain
        self.status = status
        self.ip_version = ip_version
        self.protocol = protocol
        self.source = source
        self.destination = destination
        self.enabled = enabled

    def __repr__(self):
        return '<DimensionDataFirewallRule: id=%s, name=%s, action=%s, location=%s, network_domain=%s, status=%s, ip_version=%s, protocol=%s, source=%s, destination=%s, enabled=%s>' % (self.id, self.name, self.action, self.location, self.network_domain, self.status, self.ip_version, self.protocol, self.source, self.destination, self.enabled)