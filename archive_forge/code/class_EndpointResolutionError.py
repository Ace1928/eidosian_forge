from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class EndpointResolutionError(EndpointProviderError):
    """Error when input parameters resolve to an error rule"""
    fmt = '{msg}'