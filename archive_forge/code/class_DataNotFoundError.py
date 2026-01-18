from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class DataNotFoundError(BotoCoreError):
    """
    The data associated with a particular path could not be loaded.

    :ivar data_path: The data path that the user attempted to load.
    """
    fmt = 'Unable to load data for: {data_path}'