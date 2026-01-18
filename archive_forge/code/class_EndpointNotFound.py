import inspect
import sys
from magnumclient.i18n import _
class EndpointNotFound(EndpointException):
    """Could not find requested endpoint in Service Catalog."""
    pass