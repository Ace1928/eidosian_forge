from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class EndpointVariantError(BaseEndpointResolverError):
    """
    Could not construct modeled endpoint variant.

    :ivar error_msg: The message explaining why the modeled endpoint variant
        is unable to be constructed.

    """
    fmt = 'Unable to construct a modeled endpoint with the following variant(s) {tags}: '