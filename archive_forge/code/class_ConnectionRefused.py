import inspect
import sys
from magnumclient.i18n import _
class ConnectionRefused(ConnectionError):
    """Connection refused while trying to connect to API service."""
    pass