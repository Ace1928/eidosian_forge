from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class InvalidIMDSEndpointError(BotoCoreError):
    fmt = 'Invalid endpoint EC2 Instance Metadata endpoint: {endpoint}'