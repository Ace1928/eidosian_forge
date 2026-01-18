from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnknownClientMethodError(BotoCoreError):
    """Error when trying to access a method on a client that does not exist."""
    fmt = 'Client does not have method: {method_name}'