from typing import Callable, Union, AsyncIterable, Any
from grpc.aio import (
from google.protobuf.message import Message
from sentry_sdk import Hub
from sentry_sdk.consts import OP
@staticmethod
def _update_client_call_details_metadata_from_hub(client_call_details: ClientCallDetails, hub: Hub) -> ClientCallDetails:
    metadata = list(client_call_details.metadata) if client_call_details.metadata else []
    for key, value in hub.iter_trace_propagation_headers():
        metadata.append((key, value))
    client_call_details = ClientCallDetails(method=client_call_details.method, timeout=client_call_details.timeout, metadata=metadata, credentials=client_call_details.credentials, wait_for_ready=client_call_details.wait_for_ready)
    return client_call_details