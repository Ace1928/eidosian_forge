from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnknownEndpointResolutionBuiltInName(EndpointProviderError):
    fmt = 'Unknown builtin variable name: {name}'