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
class PublisherOptions(NamedTuple):
    """The options for the publisher client.

    Attributes:
        enable_message_ordering (bool):
            Whether to order messages in a batch by a supplied ordering key.
            Defaults to false.
        flow_control (PublishFlowControl):
            Flow control settings for message publishing by the client. By default
            the publisher client does not do any throttling.
        retry (OptionalRetry):
            Retry settings for message publishing by the client. This should be
            an instance of :class:`google.api_core.retry.Retry`.
        timeout (OptionalTimeout):
            Timeout settings for message publishing by the client. It should be
            compatible with :class:`~.pubsub_v1.types.TimeoutType`.
    """
    enable_message_ordering: bool = False
    'Whether to order messages in a batch by a supplied ordering key.'
    flow_control: PublishFlowControl = PublishFlowControl()
    'Flow control settings for message publishing by the client. By default the publisher client does not do any throttling.'
    retry: 'OptionalRetry' = gapic_v1.method.DEFAULT
    'Retry settings for message publishing by the client. This should be an instance of :class:`google.api_core.retry.Retry`.'
    timeout: 'OptionalTimeout' = gapic_v1.method.DEFAULT
    'Timeout settings for message publishing by the client. It should be compatible with :class:`~.pubsub_v1.types.TimeoutType`.'