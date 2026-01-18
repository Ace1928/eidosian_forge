from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
class DataLoss(ServerError):
    """Exception mapping a :attr:`grpc.StatusCode.DATA_LOSS` error."""
    grpc_status_code = grpc.StatusCode.DATA_LOSS if grpc is not None else None