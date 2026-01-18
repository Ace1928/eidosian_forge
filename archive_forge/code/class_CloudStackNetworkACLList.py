import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackNetworkACLList:
    """
    a Network ACL for the given VPC
    """

    def __init__(self, acl_id, name, vpc_id, driver, description=None):
        """
        a Network ACL for the given VPC

        @note: This is a non-standard extension API, and only works for
               Cloudstack.

        :param      acl_id: ACL ID
        :type       acl_id: ``int``

        :param      name: Name of the network ACL List
        :type       name: ``str``

        :param      vpc_id: Id of the VPC associated with this network ACL List
        :type       vpc_id: ``string``

        :param      description: Description of the network ACL List
        :type       description: ``str``

        :rtype: :class:`CloudStackNetworkACLList`
        """
        self.id = acl_id
        self.name = name
        self.vpc_id = vpc_id
        self.driver = driver
        self.description = description

    def __repr__(self):
        return '<CloudStackNetworkACLList: id=%s, name=%s, vpc_id=%s, driver=%s, description=%s>' % (self.id, self.name, self.vpc_id, self.driver.name, self.description)