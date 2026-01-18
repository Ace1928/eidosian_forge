from libcloud.common.aws import SignedAWSConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.drivers.s3 import API_VERSION, BaseS3Connection, BaseS3StorageDriver
class MinIOStorageDriver(BaseS3StorageDriver):
    name = 'MinIO Storage Driver'
    website = 'https://min.io/'
    connectionCls = MinIOConnectionAWS4
    region_name = ''

    def __init__(self, key, secret=None, secure=True, host=None, port=None):
        if host is None:
            raise LibcloudError('host argument is required', driver=self)
        self.connectionCls.host = host
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port)