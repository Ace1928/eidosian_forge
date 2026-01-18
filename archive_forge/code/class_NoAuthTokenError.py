from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class NoAuthTokenError(BotoCoreError):
    """
    No authorization token could be found.
    """
    fmt = 'Unable to locate authorization token'