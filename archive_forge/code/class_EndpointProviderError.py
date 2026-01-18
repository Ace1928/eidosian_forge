from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class EndpointProviderError(BotoCoreError):
    """Base error for the EndpointProvider class"""
    fmt = '{msg}'