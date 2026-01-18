from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class CapacityNotAvailableError(BotoCoreError):
    fmt = 'Insufficient request capacity available.'