from libcloud.common.types import LibcloudError
from libcloud.storage.providers import Provider
from libcloud.storage.drivers.s3 import BaseS3Connection, BaseS3StorageDriver
class BaseAuroraObjectsStorageDriver(BaseS3StorageDriver):
    type = Provider.AURORAOBJECTS
    name = 'PCextreme AuroraObjects'
    website = 'https://www.pcextreme.com/aurora/objects'