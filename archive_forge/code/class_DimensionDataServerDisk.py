from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataServerDisk:
    """
    A class that represents the disk on a server
    """

    def __init__(self, id=None, scsi_id=None, size_gb=None, speed=None, state=None):
        """
        Instantiate a new :class:`DimensionDataServerDisk`

        :param id: The id of the disk
        :type  id: ``str``

        :param scsi_id: Representation for scsi
        :type  scsi_id: ``int``

        :param size_gb: Size of the disk
        :type  size_gb: ``int``

        :param speed: Speed of the disk (i.e. STANDARD)
        :type  speed: ``str``

        :param state: State of the disk (i.e. PENDING)
        :type  state: ``str``
        """
        self.id = id
        self.scsi_id = scsi_id
        self.size_gb = size_gb
        self.speed = speed
        self.state = state

    def __repr__(self):
        return '<DimensionDataServerDisk: id=%s, size_gb=%s' % (self.id, self.size_gb)