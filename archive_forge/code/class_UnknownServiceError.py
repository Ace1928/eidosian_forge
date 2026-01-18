from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
class UnknownServiceError(DataNotFoundError):
    """Raised when trying to load data for an unknown service.

    :ivar service_name: The name of the unknown service.

    """
    fmt = "Unknown service: '{service_name}'. Valid service names are: {known_service_names}"