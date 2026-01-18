from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class StubAssertionError(StubResponseError, AssertionError):
    pass