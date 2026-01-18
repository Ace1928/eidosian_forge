from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class SSOTokenLoadError(SSOError):
    fmt = 'Error loading SSO Token: {error_msg}'