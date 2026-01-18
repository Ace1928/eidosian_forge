import functools
import threading
from google.api_core import exceptions
from google.api_core import protobuf_helpers
from google.api_core.future import polling
from google.longrunning import operations_pb2
from cloudsdk.google.protobuf import json_format
from google.rpc import code_pb2
def _cancel_http(api_request, operation_name):
    """Cancel an operation using a JSON/HTTP client.

    Args:
        api_request (Callable): A callable used to make an API request. This
            should generally be
            :meth:`google.cloud._http.Connection.api_request`.
        operation_name (str): The name of the operation.
    """
    path = 'operations/{}:cancel'.format(operation_name)
    api_request(method='POST', path=path)