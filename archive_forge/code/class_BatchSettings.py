from __future__ import absolute_import
import collections
import enum
import inspect
import sys
import typing
from typing import Dict, NamedTuple, Union
import proto  # type: ignore
from google.api import http_pb2  # type: ignore
from google.api_core import gapic_v1
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2
from google.iam.v1.logging import audit_data_pb2  # type: ignore
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import duration_pb2
from cloudsdk.google.protobuf import empty_pb2
from cloudsdk.google.protobuf import field_mask_pb2
from cloudsdk.google.protobuf import timestamp_pb2
from google.api_core.protobuf_helpers import get_messages
from google.pubsub_v1.types import pubsub as pubsub_gapic_types
class BatchSettings(NamedTuple):
    """The settings for batch publishing the messages.

    Attributes:
        max_bytes (int):
            The maximum total size of the messages to collect before automatically
            publishing the batch, including any byte size overhead of the publish
            request itself. The maximum value is bound by the server-side limit of
            10_000_000 bytes. Defaults to 1 MB.
        max_latency (float):
            The maximum number of seconds to wait for additional messages before
            automatically publishing the batch. Defaults to 10ms.
        max_messages (int):
            The maximum number of messages to collect before automatically
            publishing the batch. Defaults to 100.
    """
    max_bytes: int = 1 * 1000 * 1000
    'The maximum total size of the messages to collect before automatically publishing the batch, including any byte size overhead of the publish request itself. The maximum value is bound by the server-side limit of 10_000_000 bytes.'
    max_latency: float = 0.01
    'The maximum number of seconds to wait for additional messages before automatically publishing the batch.'
    max_messages: int = 100
    'The maximum number of messages to collect before automatically publishing the batch.'