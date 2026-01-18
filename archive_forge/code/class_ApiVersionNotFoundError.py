from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class ApiVersionNotFoundError(BotoCoreError):
    """
    The data associated with either the API version or a compatible one
    could not be loaded.

    :ivar data_path: The data path that the user attempted to load.
    :ivar api_version: The API version that the user attempted to load.
    """
    fmt = 'Unable to load data {data_path} for: {api_version}'