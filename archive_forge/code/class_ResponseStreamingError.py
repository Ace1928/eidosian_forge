from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class ResponseStreamingError(HTTPClientError):
    fmt = 'An error occurred while reading from response stream: {error}'