from libcloud.compute.providers import Provider
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver
class AuroraComputeNodeDriver(CloudStackNodeDriver):
    type = Provider.AURORACOMPUTE
    name = 'PCextreme AuroraCompute'
    website = 'https://www.pcextreme.com/aurora/compute'

    def __init__(self, key, secret, path=None, host=None, url=None, region=None):
        if host is None:
            host = 'api.auroracompute.eu'
        if path is None:
            path = REGION_ENDPOINT_MAP.get(region, '/ams')
        super().__init__(key=key, secret=secret, host=host, path=path, secure=True)