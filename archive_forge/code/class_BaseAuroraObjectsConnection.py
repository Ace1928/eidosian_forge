from libcloud.common.types import LibcloudError
from libcloud.storage.providers import Provider
from libcloud.storage.drivers.s3 import BaseS3Connection, BaseS3StorageDriver
class BaseAuroraObjectsConnection(BaseS3Connection):
    host = AURORA_OBJECTS_EU_HOST