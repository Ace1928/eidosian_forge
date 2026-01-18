from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class RefreshWithMFAUnsupportedError(BotoCoreError):
    fmt = 'Cannot refresh credentials: MFA token required.'