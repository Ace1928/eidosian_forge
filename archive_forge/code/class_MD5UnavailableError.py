from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class MD5UnavailableError(BotoCoreError):
    fmt = 'This system does not support MD5 generation.'