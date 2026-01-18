from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class InvalidHostLabelError(BotoCoreError):
    """Error when an invalid host label would be bound to an endpoint"""
    fmt = 'Invalid host label to be bound to the hostname of the endpoint: "{label}".'