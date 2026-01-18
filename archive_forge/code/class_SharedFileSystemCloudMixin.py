from openstack.shared_file_system.v2._proxy import Proxy
class SharedFileSystemCloudMixin:
    share: Proxy

    def list_share_availability_zones(self):
        """List all availability zones for the Shared File Systems service.

        :returns: A list of Shared File Systems Availability Zones.
        """
        return list(self.share.availability_zones())