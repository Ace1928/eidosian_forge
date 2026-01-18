from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class InvalidRetryConfigurationError(BotoCoreError):
    """Error when invalid retry configuration is specified"""
    fmt = 'Cannot provide retry configuration for "{retry_config_option}". Valid retry configuration options are: {valid_options}'