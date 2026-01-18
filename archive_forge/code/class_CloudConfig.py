from openstack.config import cloud_region
class CloudConfig(cloud_region.CloudRegion):

    def __init__(self, name, region, config, **kwargs):
        super(CloudConfig, self).__init__(name, region, config, **kwargs)
        self.region = region