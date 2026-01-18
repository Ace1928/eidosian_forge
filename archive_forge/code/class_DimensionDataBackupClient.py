from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataBackupClient:
    """
    An object that represents a backup client
    """

    def __init__(self, id, type, status, schedule_policy, storage_policy, download_url, alert=None, running_job=None):
        """
        Initialize an instance of :class:`DimensionDataBackupClient`

        :param id: Unique ID for the client
        :type  id: ``str``

        :param type: The type of client that this client is
        :type  type: :class:`DimensionDataBackupClientType`

        :param status: The states of this particular backup client.
                       i.e. (Unregistered)
        :type  status: ``str``

        :param schedule_policy: The schedule policy for this client
                                NOTE: Dimension Data only sends back the name
                                of the schedule policy, no further details
        :type  schedule_policy: ``str``

        :param storage_policy: The storage policy for this client
                               NOTE: Dimension Data only sends back the name
                               of the storage policy, no further details
        :type  storage_policy: ``str``

        :param download_url: The download url for this client
        :type  download_url: ``str``

        :param alert: The alert configured for this backup client (optional)
        :type  alert: :class:`DimensionDataBackupClientAlert`

        :param alert: The running job for the client (optional)
        :type  alert: :class:`DimensionDataBackupClientRunningJob`
        """
        self.id = id
        self.type = type
        self.status = status
        self.schedule_policy = schedule_policy
        self.storage_policy = storage_policy
        self.download_url = download_url
        self.alert = alert
        self.running_job = running_job

    def __repr__(self):
        return '<DimensionDataBackupClient: id=%s>' % self.id