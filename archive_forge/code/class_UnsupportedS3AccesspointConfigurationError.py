from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnsupportedS3AccesspointConfigurationError(BotoCoreError):
    """Error when an unsupported configuration is used with access-points"""
    fmt = 'Unsupported configuration when using S3 access-points: {msg}'