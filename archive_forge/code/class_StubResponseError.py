from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class StubResponseError(BotoCoreError):
    fmt = 'Error getting response stub for operation {operation_name}: {reason}'