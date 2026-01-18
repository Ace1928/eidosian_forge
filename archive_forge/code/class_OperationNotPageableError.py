from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class OperationNotPageableError(BotoCoreError):
    fmt = 'Operation cannot be paginated: {operation_name}'