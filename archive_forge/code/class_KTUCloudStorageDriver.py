from libcloud.common.types import LibcloudError
from libcloud.storage.providers import Provider
from libcloud.storage.drivers.cloudfiles import CloudFilesConnection, CloudFilesStorageDriver
class KTUCloudStorageDriver(CloudFilesStorageDriver):
    """
    Cloudfiles storage driver for the UK endpoint.
    """
    type = Provider.KTUCLOUD
    name = 'KTUCloud Storage'
    connectionCls = KTUCloudStorageConnection