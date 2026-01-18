from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class ConfigNotFound(BotoCoreError):
    """
    The specified configuration file could not be found.

    :ivar path: The path to the configuration file.
    """
    fmt = 'The specified config file ({path}) could not be found.'