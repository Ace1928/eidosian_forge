from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
class MethodNotImplemented(ServerError):
    """Exception mapping a ``501 Not Implemented`` response or a
    :attr:`grpc.StatusCode.UNIMPLEMENTED` error."""
    code = http.client.NOT_IMPLEMENTED
    grpc_status_code = grpc.StatusCode.UNIMPLEMENTED if grpc is not None else None