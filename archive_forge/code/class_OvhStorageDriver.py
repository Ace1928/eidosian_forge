from libcloud.storage.drivers.s3 import (
class OvhStorageDriver(BaseS3StorageDriver):
    name = 'Ovh Storage Driver'
    website = 'https://www.ovhcloud.com/en/public-cloud/object-storage/'
    connectionCls = S3SignatureV4Connection
    region_name = 'sbg'

    def __init__(self, key, secret=None, secure=True, host=None, port=None, region='sbg', url=None, **kwargs):
        if hasattr(self, 'region_name') and (not region):
            region = self.region_name
        self.region_name = region
        if region and region not in REGION_TO_HOST_MAP.keys():
            raise ValueError('Invalid or unsupported region: %s' % region)
        self.name = 'Ovh Object Storage (%s)' % region
        if host is None:
            self.connectionCls.host = REGION_TO_HOST_MAP[region]
        else:
            self.connectionCls.host = host
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, region=region, url=url, **kwargs)

    @classmethod
    def list_regions(self):
        return REGION_TO_HOST_MAP.keys()

    def get_object_cdn_url(self, obj, ex_expiry=S3_CDN_URL_EXPIRY_HOURS):
        return S3StorageDriver.get_object_cdn_url(self, obj, ex_expiry=ex_expiry)