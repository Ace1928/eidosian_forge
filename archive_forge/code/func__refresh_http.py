import functools
import threading
from google.api_core import exceptions
from google.api_core import protobuf_helpers
from google.api_core.future import polling
from google.longrunning import operations_pb2
from cloudsdk.google.protobuf import json_format
from google.rpc import code_pb2
def _refresh_http(api_request, operation_name, retry=None):
    """Refresh an operation using a JSON/HTTP client.

    Args:
        api_request (Callable): A callable used to make an API request. This
            should generally be
            :meth:`google.cloud._http.Connection.api_request`.
        operation_name (str): The name of the operation.
        retry (google.api_core.retry.Retry): (Optional) retry policy

    Returns:
        google.longrunning.operations_pb2.Operation: The operation.
    """
    path = 'operations/{}'.format(operation_name)
    if retry is not None:
        api_request = retry(api_request)
    api_response = api_request(method='GET', path=path)
    return json_format.ParseDict(api_response, operations_pb2.Operation())