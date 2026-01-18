from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnknownKeyError(ValidationError):
    """
    Unknown key in a struct parameter.

    :ivar value: The value that was being checked.
    :ivar param: The name of the parameter.
    :ivar choices: The valid choices the value can be.
    """
    fmt = "Unknown key '{value}' for param '{param}'.  Must be one of: {choices}"