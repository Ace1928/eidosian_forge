import collections
import grpc
from google.api_core import exceptions
from google.api_core import retry
from google.api_core import timeout
def _exception_class_for_grpc_status_name(name):
    """Returns the Google API exception class for a gRPC error code name.

    DEPRECATED: use ``exceptions.exception_class_for_grpc_status`` method
    directly instead.

    Args:
        name (str): The name of the gRPC status code, for example,
            ``UNAVAILABLE``.

    Returns:
        :func:`type`: The appropriate subclass of
            :class:`google.api_core.exceptions.GoogleAPICallError`.
    """
    return exceptions.exception_class_for_grpc_status(getattr(grpc.StatusCode, name))