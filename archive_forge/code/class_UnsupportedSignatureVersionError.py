from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnsupportedSignatureVersionError(BotoCoreError):
    """Error when trying to use an unsupported Signature Version."""
    fmt = 'Signature version is not supported: {signature_version}'