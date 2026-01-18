from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class MetadataRetrievalError(BotoCoreError):
    fmt = 'Error retrieving metadata: {error_msg}'