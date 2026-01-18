import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
class EndpointDiscoveryRequired(EndpointDiscoveryException):
    """Endpoint Discovery is disabled but is required for this operation."""
    fmt = 'Endpoint Discovery is not enabled but this operation requires it.'