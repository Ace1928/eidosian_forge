from __future__ import division
import collections
import functools
import itertools
import logging
import threading
import typing
from typing import Any, Dict, Callable, Iterable, List, Optional, Set, Tuple
import uuid
import grpc  # type: ignore
from google.api_core import bidi
from google.api_core import exceptions
from google.cloud.pubsub_v1 import types
from google.cloud.pubsub_v1.subscriber._protocol import dispatcher
from google.cloud.pubsub_v1.subscriber._protocol import heartbeater
from google.cloud.pubsub_v1.subscriber._protocol import histogram
from google.cloud.pubsub_v1.subscriber._protocol import leaser
from google.cloud.pubsub_v1.subscriber._protocol import messages_on_hold
from google.cloud.pubsub_v1.subscriber._protocol import requests
from google.cloud.pubsub_v1.subscriber.exceptions import (
import google.cloud.pubsub_v1.subscriber.message
from google.cloud.pubsub_v1.subscriber import futures
from google.cloud.pubsub_v1.subscriber.scheduler import ThreadScheduler
from google.pubsub_v1 import types as gapic_types
from google.rpc.error_details_pb2 import ErrorInfo  # type: ignore
from google.rpc import code_pb2  # type: ignore
from google.rpc import status_pb2
def _process_requests(error_status: Optional['status_pb2.Status'], ack_reqs_dict: Dict[str, requests.AckRequest], errors_dict: Optional[Dict[str, str]]):
    """Process requests when exactly-once delivery is enabled by referring to
    error_status and errors_dict.

    The errors returned by the server in as `error_status` or in `errors_dict`
    are used to complete the request futures in `ack_reqs_dict` (with a success
    or exception) or to return requests for further retries.
    """
    requests_completed = []
    requests_to_retry = []
    for ack_id in ack_reqs_dict:
        if errors_dict and ack_id in errors_dict:
            exactly_once_error = errors_dict[ack_id]
            if exactly_once_error.startswith('TRANSIENT_'):
                requests_to_retry.append(ack_reqs_dict[ack_id])
            else:
                if exactly_once_error == 'PERMANENT_FAILURE_INVALID_ACK_ID':
                    exc = AcknowledgeError(AcknowledgeStatus.INVALID_ACK_ID, info=None)
                else:
                    exc = AcknowledgeError(AcknowledgeStatus.OTHER, exactly_once_error)
                future = ack_reqs_dict[ack_id].future
                if future is not None:
                    future.set_exception(exc)
                requests_completed.append(ack_reqs_dict[ack_id])
        elif error_status and error_status.code in _EXACTLY_ONCE_DELIVERY_TEMPORARY_RETRY_ERRORS:
            requests_to_retry.append(ack_reqs_dict[ack_id])
        elif error_status:
            if error_status.code == code_pb2.PERMISSION_DENIED:
                exc = AcknowledgeError(AcknowledgeStatus.PERMISSION_DENIED, info=None)
            elif error_status.code == code_pb2.FAILED_PRECONDITION:
                exc = AcknowledgeError(AcknowledgeStatus.FAILED_PRECONDITION, info=None)
            else:
                exc = AcknowledgeError(AcknowledgeStatus.OTHER, str(error_status))
            future = ack_reqs_dict[ack_id].future
            if future is not None:
                future.set_exception(exc)
            requests_completed.append(ack_reqs_dict[ack_id])
        elif ack_reqs_dict[ack_id].future:
            future = ack_reqs_dict[ack_id].future
            assert future is not None
            future.set_result(AcknowledgeStatus.SUCCESS)
            requests_completed.append(ack_reqs_dict[ack_id])
        else:
            requests_completed.append(ack_reqs_dict[ack_id])
    return (requests_completed, requests_to_retry)