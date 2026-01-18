from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class ParamValidationError(BotoCoreError):
    fmt = 'Parameter validation failed:\n{report}'