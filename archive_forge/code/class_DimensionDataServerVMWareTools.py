from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataServerVMWareTools:
    """
    A class that represents the VMWareTools for a node
    """

    def __init__(self, status, version_status, api_version):
        """
        Instantiate a new :class:`DimensionDataServerVMWareTools` object

        :param status: The status of VMWare Tools
        :type  status: ``str``

        :param version_status: The status for the version of VMWare Tools
            (i.e NEEDS_UPGRADE)
        :type  version_status: ``str``

        :param api_version: The API version of VMWare Tools
        :type  api_version: ``str``
        """
        self.status = status
        self.version_status = version_status
        self.api_version = api_version

    def __repr__(self):
        return '<DimensionDataServerVMWareTools status=%s, version_status=%s, api_version=%s>' % (self.status, self.version_status, self.api_version)