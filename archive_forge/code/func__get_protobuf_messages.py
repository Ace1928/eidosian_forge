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
def _get_protobuf_messages(module: 'ModuleType') -> Dict[str, proto.Message]:
    """Discover all protobuf Message classes in a given import module.

    Args:
        module (module): A Python module; :func:`dir` will be run against this
            module to find Message subclasses.

    Returns:
        dict[str, proto.Message]: A dictionary with the
            Message class names as keys, and the Message subclasses themselves
            as values.
    """
    answer = collections.OrderedDict()
    for name in dir(module):
        candidate = getattr(module, name)
        if inspect.isclass(candidate) and issubclass(candidate, proto.Message):
            answer[name] = candidate
    return answer