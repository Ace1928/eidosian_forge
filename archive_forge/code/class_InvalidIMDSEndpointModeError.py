from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class InvalidIMDSEndpointModeError(BotoCoreError):
    fmt = 'Invalid EC2 Instance Metadata endpoint mode: {mode} Valid endpoint modes (case-insensitive): {valid_modes}.'