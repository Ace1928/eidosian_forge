from libcloud.common.types import LibcloudError
from libcloud.storage.providers import Provider
from libcloud.storage.drivers.s3 import BaseS3Connection, BaseS3StorageDriver
class AuroraObjectsStorageDriver(BaseAuroraObjectsStorageDriver):
    connectionCls = BaseAuroraObjectsConnection

    def enable_container_cdn(self, *argv):
        raise LibcloudError(NO_CDN_SUPPORT_ERROR, driver=self)

    def enable_object_cdn(self, *argv):
        raise LibcloudError(NO_CDN_SUPPORT_ERROR, driver=self)

    def get_container_cdn_url(self, *argv):
        raise LibcloudError(NO_CDN_SUPPORT_ERROR, driver=self)

    def get_object_cdn_url(self, *argv):
        raise LibcloudError(NO_CDN_SUPPORT_ERROR, driver=self)