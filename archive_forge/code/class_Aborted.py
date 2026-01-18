from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
class Aborted(Conflict):
    """Exception mapping a :attr:`grpc.StatusCode.ABORTED` error."""
    grpc_status_code = grpc.StatusCode.ABORTED if grpc is not None else None