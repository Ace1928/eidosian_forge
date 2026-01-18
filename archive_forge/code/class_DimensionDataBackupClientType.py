from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataBackupClientType:
    """
    A client type object for backups
    """

    def __init__(self, type, is_file_system, description):
        """
        Initialize an instance of :class:`DimensionDataBackupClientType`

        :param type: The type of client i.e. (FA.Linux, MySQL, etc.)
        :type  type: ``str``

        :param is_file_system: The name of the iRule
        :type  is_file_system: ``bool``

        :param description: Description of the client
        :type  description: ``str``
        """
        self.type = type
        self.is_file_system = is_file_system
        self.description = description

    def __repr__(self):
        return '<DimensionDataBackupClientType: type=%s>' % self.type