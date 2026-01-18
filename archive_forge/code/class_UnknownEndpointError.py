from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnknownEndpointError(BaseEndpointResolverError, ValueError):
    """
    Could not construct an endpoint.

    :ivar service_name: The name of the service.
    :ivar region_name: The name of the region.
    """
    fmt = 'Unable to construct an endpoint for {service_name} in region {region_name}'