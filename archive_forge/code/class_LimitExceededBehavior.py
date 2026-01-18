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
class LimitExceededBehavior(str, enum.Enum):
    """The possible actions when exceeding the publish flow control limits."""
    IGNORE = 'ignore'
    BLOCK = 'block'
    ERROR = 'error'