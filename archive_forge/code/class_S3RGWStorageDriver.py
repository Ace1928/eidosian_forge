from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, SignedAWSConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.drivers.s3 import (
class S3RGWStorageDriver(BaseS3StorageDriver):
    name = 'Ceph RGW'
    website = 'http://ceph.com/'

    def __init__(self, key, secret=None, secure=True, host=None, port=None, api_version=None, region=S3_RGW_DEFAULT_REGION, **kwargs):
        if host is None:
            raise LibcloudError('host required', driver=self)
        self.name = kwargs.pop('name', None)
        if self.name is None:
            self.name = 'Ceph RGW S3 (%s)' % region
        self.ex_location_name = region
        self.region_name = region
        self.signature_version = str(kwargs.pop('signature_version', DEFAULT_SIGNATURE_VERSION))
        if self.signature_version not in ['2', '4']:
            raise ValueError('Invalid signature_version: %s' % self.signature_version)
        if self.signature_version == '2':
            self.connectionCls = S3RGWConnectionAWS2
        elif self.signature_version == '4':
            self.connectionCls = S3RGWConnectionAWS4
        self.connectionCls.host = host
        super().__init__(key, secret, secure, host, port, api_version, region, **kwargs)

    def _ex_connection_class_kwargs(self):
        kwargs = {}
        kwargs['signature_version'] = self.signature_version
        return kwargs