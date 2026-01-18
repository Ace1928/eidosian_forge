from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
class RequestRangeNotSatisfiable(ClientError):
    """Exception mapping a ``416 Request Range Not Satisfiable`` response."""
    code = http.client.REQUESTED_RANGE_NOT_SATISFIABLE