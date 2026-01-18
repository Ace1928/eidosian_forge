from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class InvalidMaxRetryAttemptsError(InvalidRetryConfigurationError):
    """Error when invalid retry configuration is specified"""
    fmt = 'Value provided to "max_attempts": {provided_max_attempts} must be an integer greater than or equal to {min_value}.'