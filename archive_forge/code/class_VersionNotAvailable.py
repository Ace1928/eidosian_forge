import os_service_types
from keystoneauth1.exceptions import base
class VersionNotAvailable(DiscoveryFailure):
    message = 'Discovery failed. Requested version is not available.'