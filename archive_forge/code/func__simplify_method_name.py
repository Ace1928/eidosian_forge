from typing import Generic, Iterator, Optional, TypeVar
import collections
import functools
import warnings
import grpc
from google.api_core import exceptions
import google.auth
import google.auth.credentials
import google.auth.transport.grpc
import google.auth.transport.requests
import cloudsdk.google.protobuf
def _simplify_method_name(method):
    """Simplifies a gRPC method name.

    When gRPC invokes the channel to create a callable, it gives a full
    method name like "/google.pubsub.v1.Publisher/CreateTopic". This
    returns just the name of the method, in this case "CreateTopic".

    Args:
        method (str): The name of the method.

    Returns:
        str: The simplified name of the method.
    """
    return method.rsplit('/', 1).pop()