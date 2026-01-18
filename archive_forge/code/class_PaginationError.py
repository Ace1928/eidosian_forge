from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class PaginationError(BotoCoreError):
    fmt = 'Error during pagination: {message}'