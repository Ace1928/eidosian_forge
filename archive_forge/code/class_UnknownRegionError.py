from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnknownRegionError(BotoCoreError):
    """Raised when trying to load data for an unknown region.

    :ivar region_name: The name of the unknown region.

    """
    fmt = "Unknown region: '{region_name}'. {error_msg}"